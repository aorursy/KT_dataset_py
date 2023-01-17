import os

import shutil



from glob   import glob

from PIL    import Image



import pickle



import numpy  as np

import pandas as pd



from torch                  import cat, no_grad

from torch.utils.data       import DataLoader

from torchvision.transforms import ToTensor, Grayscale, Resize, Compose
# Image dimensions

HEIGHT = 137

WIDTH  = 236



# Bounding Box

def bbox(image):

    """

    Determines the bounding boxes for images to remove empty space where possible

    :param image:

    :return:

    """

    for i in range(image.shape[1]):

        if not np.all(image[:,i] == 0):

            x_min = i - 13 if (i > 13) else 0

            break



    for i in reversed(range(image.shape[1])):

        if not np.all(image[:,i] == 0):

            x_max = i + 13 if (i < WIDTH - 13) else WIDTH

            break



    for i in range(image.shape[0]):

        if not np.all(image[i] == 0):

            y_min = i - 10 if (i > 10) else 0

            break



    for i in reversed(range(image.shape[0])):

        if not np.all(image[i] == 0):

            y_max = i + 10 if (i < HEIGHT - 10) else HEIGHT

            break



    return x_min, y_min, x_max, y_max



# Create a folder

os.makedirs("images")



# Convert the files to images

for fl in glob("../input/bengaliai-cv19/test_image_data_*.parquet"):

    # Load the dataset

    df = pd.read_parquet(fl)



    # Fetch the images and IDs

    ids, images = df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)

    del(df)



    # Save each image

    for id,image in zip(ids, images):

        # Input data is inverted

        image_ = 255 - image



        # Remove the boundaries of the image

        image_ = image_[5:-5, 5:-5]



        # Apply max normalization

        image_ = (image_ * (255.0 / image_.max())).astype(np.uint8)



        # Filter low-intensity pixels

        image_[image_ < 50]  = 0

        #image_[image_ >= 100] = 255



        # Crop the image

        x_min, y_min, x_max, y_max = bbox(image_)

        image_ = image_[y_min:y_max, x_min:x_max]



        image_ = Image.fromarray(image_)



        # Save

        image_.save("images/{}.png".format(id))

class Dataset(object):

    def __init__(self, image_folder:str):

        """

        Class for creating a dataset for model inference

        """

        # Class attributes

        self.image_list = glob(image_folder + "/*.png")

        self.transform  = Compose([ Grayscale(num_output_channels=1)

                                  , Resize(size=(128, 128))

                                  , ToTensor()

                                  ]

                                 )



    def __len__(self):

        return len(self.image_list)



    def __getitem__(self, idx):

        # Load the image

        img = Image.open(self.image_list[idx])



        # Convert to a tensor

        img = self.transform(img)



        return img



# Create a dataset

ds = Dataset(image_folder="images")



# Create a loader

loader = DataLoader(ds, batch_size=512, shuffle=False)
from collections import OrderedDict

import math



import torch

import torch.nn as nn

import torch.nn.functional as F



#%% Model class

class SeResNeXt101(nn.Module):

    def __init__( self

                , n_class    =1001

                , input_shape=[1, 128, 128]

                ):

        super(SeResNeXt101, self).__init__()



        self.model = SENet( in_channels           =input_shape[0]

                          , block                 =SEResNeXtBottleneck

                          , layers                =[3, 4, 23, 3]

                          , groups                =32

                          , reduction             =16

                          , dropout_p             =None

                          , inplanes              =64

                          , input_3x3             =False

                          , downsample_kernel_size=1

                          , downsample_padding    =0

                          , num_classes           =n_class

                          )



    def forward(self, x):

        x = self.model(x)



        return x





class SEModule(nn.Module):

    def __init__(self, channels, reduction):

        super(SEModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,

                             padding=0)

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,

                             padding=0)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):

        module_input = x

        x = self.avg_pool(x)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.fc2(x)

        x = self.sigmoid(x)

        return module_input * x





class Bottleneck(nn.Module):

    """

    Base class for bottlenecks that implements `forward()` method.

    """

    def forward(self, x):

        residual = x



        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)



        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)



        out = self.conv3(out)

        out = self.bn3(out)



        if self.downsample is not None:

            residual = self.downsample(x)



        out = self.se_module(out) + residual

        out = self.relu(out)



        return out





class SEBottleneck(Bottleneck):

    """

    Bottleneck for SENet154.

    """

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None):

        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,

                               stride=stride, padding=1, groups=groups,

                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes * 4)

        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,

                               bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SEResNetBottleneck(Bottleneck):

    """

    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe

    implementation and uses `stride=stride` in `conv1` and not in `conv2`

    (the latter is used in the torchvision implementation of ResNet).

    """

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None):

        super(SEResNetBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,

                               stride=stride)

        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,

                               groups=groups, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SEResNeXtBottleneck(Bottleneck):

    """

    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.

    """

    expansion = 4



    def __init__(self, inplanes, planes, groups, reduction, stride=1,

                 downsample=None, base_width=4):

        super(SEResNeXtBottleneck, self).__init__()

        width = math.floor(planes * (base_width / 64)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,

                               stride=1)

        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,

                               padding=1, groups=groups, bias=False)

        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.LeakyReLU(negative_slope=0.1)

        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample

        self.stride = stride





class SENet(nn.Module):

    def __init__(self, in_channels, block, layers, groups, reduction, dropout_p=0.2,

                 inplanes=128, input_3x3=True, downsample_kernel_size=3,

                 downsample_padding=1, num_classes=1000):

        """

        Parameters

        ----------

        block (nn.Module): Bottleneck class.

            - For SENet154: SEBottleneck

            - For SE-ResNet models: SEResNetBottleneck

            - For SE-ResNeXt models:  SEResNeXtBottleneck

        layers (list of ints): Number of residual blocks for 4 layers of the

            network (layer1...layer4).

        groups (int): Number of groups for the 3x3 convolution in each

            bottleneck block.

            - For SENet154: 64

            - For SE-ResNet models: 1

            - For SE-ResNeXt models:  32

        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.

            - For all models: 16

        dropout_p (float or None): Drop probability for the Dropout layer.

            If `None` the Dropout layer is not used.

            - For SENet154: 0.2

            - For SE-ResNet models: None

            - For SE-ResNeXt models: None

        inplanes (int):  Number of input channels for layer1.

            - For SENet154: 128

            - For SE-ResNet models: 64

            - For SE-ResNeXt models: 64

        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of

            a single 7x7 convolution in layer0.

            - For SENet154: True

            - For SE-ResNet models: False

            - For SE-ResNeXt models: False

        downsample_kernel_size (int): Kernel size for downsampling convolutions

            in layer2, layer3 and layer4.

            - For SENet154: 3

            - For SE-ResNet models: 1

            - For SE-ResNeXt models: 1

        downsample_padding (int): Padding for downsampling convolutions in

            layer2, layer3 and layer4.

            - For SENet154: 1

            - For SE-ResNet models: 0

            - For SE-ResNeXt models: 0

        num_classes (int): Number of outputs in `last_linear` layer.

            - For all models: 1000

        """

        super(SENet, self).__init__()

        self.inplanes = inplanes

        if input_3x3:

            layer0_modules = [

                ('conv1', nn.Conv2d(in_channels, 64, 3, stride=2, padding=1,

                                    bias=False)),

                ('bn1', nn.BatchNorm2d(64)),

                ('relu1', nn.LeakyReLU(negative_slope=0.1)),

                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn2', nn.BatchNorm2d(64)),

                ('relu2', nn.LeakyReLU(negative_slope=0.1)),

                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,

                                    bias=False)),

                ('bn3', nn.BatchNorm2d(inplanes)),

                ('relu3', nn.LeakyReLU(negative_slope=0.1)),

            ]

        else:

            layer0_modules = [

                ('conv1', nn.Conv2d(in_channels, inplanes, kernel_size=7, stride=2,

                                    padding=3, bias=False)),

                ('bn1', nn.BatchNorm2d(inplanes)),

                ('relu1', nn.LeakyReLU(negative_slope=0.1)),

            ]

        # To preserve compatibility with Caffe weights `ceil_mode=True`

        # is used instead of `padding=1`.

        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,

                                                    ceil_mode=True)))

        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))

        self.layer1 = self._make_layer(

            block,

            planes=64,

            blocks=layers[0],

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=1,

            downsample_padding=0

        )

        self.layer2 = self._make_layer(

            block,

            planes=128,

            blocks=layers[1],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer3 = self._make_layer(

            block,

            planes=256,

            blocks=layers[2],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.layer4 = self._make_layer(

            block,

            planes=512,

            blocks=layers[3],

            stride=2,

            groups=groups,

            reduction=reduction,

            downsample_kernel_size=downsample_kernel_size,

            downsample_padding=downsample_padding

        )

        self.avg_pool = nn.AvgPool2d(4, stride=1)

        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        self.last_linear = nn.Linear(512 * block.expansion, num_classes)



    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,

                    downsample_kernel_size=1, downsample_padding=0):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(

                nn.Conv2d(self.inplanes, planes * block.expansion,

                          kernel_size=downsample_kernel_size, stride=stride,

                          padding=downsample_padding, bias=False),

                nn.BatchNorm2d(planes * block.expansion),

            )



        layers = []

        layers.append(block(self.inplanes, planes, groups, reduction, stride,

                            downsample))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, groups, reduction))



        return nn.Sequential(*layers)



    def features(self, x):

        x = self.layer0(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x



    def logits(self, x):

        x = self.avg_pool(x)

        if self.dropout is not None:

            x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.last_linear(x)

        return x



    def forward(self, x):

        x = self.features(x)

        x = self.logits(x)

        return x
class MobileNetV2(nn.Module):

    def __init__( self

                , n_class    =1000

                , input_shape=[1,224,224]

                , width_mult =1.0

                , name       ="MobileNetV2 Model"

                ):

        """

        TODO: Update documentation

        :param n_class:

        :param input_shape:

        :param width_mult:

        :param name:

        """

        super(MobileNetV2, self).__init__()



        # Class attributes

        self.name = name



        # Network architecture

        interverted_residual_setting = [

            # t, c, n, s

            [1, 16, 1, 1],

            [6, 24, 2, 2],

            [6, 32, 3, 2],

            [6, 64, 4, 2],

            [6, 96, 3, 1],

            [6, 160, 3, 2],

            [6, 320, 1, 1],

        ]



        # Assertions

        assert input_shape[1] % 32 == 0



        # Create the first layer

        self.layers = nn.ModuleList([Conv2D_BN(in_features=input_shape[0], out_features=32, stride=2)])



        # Build the residual blocks

        in_channels = self.layers[0].conv.out_channels

        for t, c, n, s in interverted_residual_setting:

            out_channels = make_divisible(c * width_mult) if t > 1 else c

            for i in range(n):

                if i == 0:

                    self.layers.append(InvertedResidual( in_features =in_channels

                                                       , out_features=out_channels

                                                       , stride      =s

                                                       , expand_ratio=t

                                                       )

                                     )

                else:

                    self.layers.append(InvertedResidual( in_features =in_channels

                                                       , out_features=out_channels

                                                       , stride      =1

                                                       , expand_ratio=t

                                                       )

                                      )



                in_channels = out_channels



        # Build the remaining convolutional layer

        self.layers.append(Conv2D_BN( in_features =in_channels

                                    , out_features=make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280

                                    , kernel_size =1

                                    , stride      =1

                                    , padding     =0

                                    )

                          )



        # Add the classifier

        self.layers.append(nn.Linear( in_features =self.layers[-1].conv.out_channels

                                    , out_features=n_class

                                    )

                          )



        # Initialize weights

        self._init_weight()



    # Forward function

    def forward(self, x):

        # Pass through the first layer

        for layer in self.layers[:-1]:

            x = layer(x)



        # Average pooling

        x = x.mean(3).mean(2)



        # Classification

        x = self.layers[-1](x)



        return x



    # Weight initialization

    def _init_weight(self):

        for m in self.modules():

            #print(m)

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1.0)

                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                m.weight.data.normal_(0, 0.01)

                m.bias.data.zero_()





#%% Utilities

class Conv2D_BN(nn.Module):

    def __init__(self, in_features, out_features, kernel_size=3, stride=1, padding=1, groups=1, activation=True):

        """

        Convolution followed by Batch Normalization, with optional Mish activation

        :param in_features:

        :param out_features:

        :param kernel_size:

        :param stride:

        :param padding:

        :param groups:

        :param activation:

        """

        super(Conv2D_BN, self).__init__()



        # Class parameters

        self.activation = activation



        # Convolution and BatchNorm class

        self.conv = nn.Conv2d( in_channels =in_features

                             , out_channels=out_features

                             , kernel_size =kernel_size

                             , stride      =stride

                             , padding     =padding

                             , groups      =groups

                             , bias        =False

                             )



        self.bn = nn.BatchNorm2d(num_features=out_features, momentum=0.9)



    def forward(self, x):

        x = self.conv(x)

        x = self.bn(x)



        if self.activation:

            x = F.leaky_relu(x, negative_slope=0.1)



        return x





def make_divisible(x, divisible_by=8):

    return int(np.ceil(x * 1. / divisible_by) * divisible_by)





class InvertedResidual(nn.Module):

    def __init__( self

                , in_features

                , out_features

                , stride

                , expand_ratio

                ):

        """

        MobileNet Inverted Residual block

        :param in_features:

        :param out_features:

        :param stride:

        :param expand_ratio:

        """

        super(InvertedResidual, self).__init__()



        # Input validation

        assert stride in [1, 2]



        # Class parameter

        self.stride = stride



        # Calculate the hidden dimension

        hidden_dim = int(in_features * expand_ratio)



        # Determine whether to use a residual connection

        self.res_connect = self.stride == 1 and in_features == out_features



        if expand_ratio == 1:

            conv_1 = Conv2D_BN( in_features =hidden_dim

                              , out_features=hidden_dim

                              , kernel_size =3

                              , stride      =stride

                              , padding     =1

                              , groups      =hidden_dim

                              )



            conv_2 = Conv2D_BN( in_features =hidden_dim

                              , out_features=out_features

                              , kernel_size =1

                              , stride      =1

                              , padding     =0

                              , activation  =False

                              )



            self.layers = nn.ModuleList([conv_1, conv_2])



        else:

            conv_1 = Conv2D_BN(in_features  =in_features

                              , out_features=hidden_dim

                              , kernel_size =1

                              , stride      =1

                              , padding     =0

                              )



            conv_2 = Conv2D_BN( in_features =hidden_dim

                              , out_features=hidden_dim

                              , kernel_size =3

                              , stride      =stride

                              , padding     =1

                              , groups      =hidden_dim

                              )



            conv_3 = Conv2D_BN( in_features =hidden_dim

                              , out_features=out_features

                              , kernel_size =1

                              , stride      =1

                              , padding     =0

                              , activation  =False

                              )



            self.layers = nn.ModuleList([conv_1, conv_2, conv_3])



    def forward(self, x):

        if self.res_connect:

            return x + self._block_forward(x, self.layers)

        else:

            return self._block_forward(x, self.layers)



    @staticmethod

    def _block_forward(x, block):

        for layer in block:

            x = layer(x)



        return x
# Load the models

model_root = "../input/handwritten-grapheme-models/"

models = {}
for name,checkpoint,model,classes in zip( ["root", "vowel", "consonant"]

                                        , [ model_root + "root.pth"

                                          , model_root + "vowel.pth"

                                          , model_root + "consonant.pth"

                                          ]

                                        , [SeResNeXt101, MobileNetV2, MobileNetV2]

                                        , [168, 11, 7]

                                        ):

    # Instantiate a model

    net = model(n_class=classes, input_shape=[1, 128, 128])



    # Load the state dict

    net.load_state_dict(torch.load(checkpoint))

    

    models[name] = net
# Predict

predictions = {}

for feature,model in models.items():

    model.cuda()

    model.eval()

    with no_grad():

        predictions[feature] = cat([model(x.cuda()).cpu() for x in loader]).softmax(dim=-1).argmax(dim=-1).numpy()

# Output

ids    = []

labels = []

orders = []

for i,image in enumerate(ds.image_list):

    j = int(image.split(".")[0].split("_")[-1])

    

    ids.append("Test_{}_grapheme_root".format(j))

    ids.append("Test_{}_vowel_diacritic".format(j))

    ids.append("Test_{}_consonant_diacritic".format(j))



    labels.append(predictions["root"][i])

    labels.append(predictions["vowel"][i])

    labels.append(predictions["consonant"][i])

    

    orders.append(j)

    orders.append(j + 0.3)

    orders.append(j + 0.6)



# Create a submission DataFrame

submission_df = pd.DataFrame({"row_id":ids, "target":labels, "order":orders})



# Reorder to be safe

submission_df.sort_values("order", inplace=True)

# Write output

submission_df[["row_id", "target"]].to_csv('submission.csv', index=False)



# Clean up

shutil.rmtree("images")