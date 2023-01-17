import glob

import torch.nn as nn

from albumentations.torch import ToTensor

from albumentations import Normalize, Compose

import cv2

from ipdb import set_trace as bp

import numpy as np

import os

import functools

import torch.utils.model_zoo as model_zoo

from torchvision.models.resnet import ResNet

from torchvision.models.resnet import BasicBlock

from torchvision.models.resnet import Bottleneck

# from pretrainedmodels.models.torchvision_models import pretrained_settings

import torch

import torch.nn as nn

import torch.nn.functional as F

import pandas as pd

from torch.utils.data import DataLoader, Dataset



def preprocess_input(x, mean=None, std=None, input_space='RGB', input_range=None, **kwargs):



    if input_space == 'BGR':

        x = x[..., ::-1].copy()



    if input_range is not None:

        if x.max() > 1 and input_range[1] == 1:

            x = x / 255.



    if mean is not None:

        mean = np.array(mean)

        x = x - mean



    if std is not None:

        std = np.array(std)

        x = x / std



    return x



def post_process(probability, threshold, min_size):

    '''Post processing of each predicted mask, components with lesser number of pixels

    than `min_size` are ignored'''



    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]

    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predictions = np.zeros((256, 1600), np.float32)

    num = 0

    for c in range(1, num_component):

        p = (component == c)

        if p.sum() > min_size:

            predictions[p] = 1

            num += 1

    return predictions, num





class Model(nn.Module):



    def __init__(self):

        super().__init__()



    def initialize(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)





class Conv2dReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,

                 stride=1, use_batchnorm=True, **batchnorm_params):



        super().__init__()



        layers = [

            nn.Conv2d(in_channels, out_channels, kernel_size,

                              stride=stride, padding=padding, bias=not (use_batchnorm)),

            nn.ReLU(inplace=True),

        ]



        if use_batchnorm:

            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))



        self.block = nn.Sequential(*layers)



    def forward(self, x):

        return self.block(x)





class EncoderDecoder(Model):



    def __init__(self, encoder, decoder, activation):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder



        if callable(activation) or activation is None:

            self.activation = activation

        elif activation == 'softmax':

            self.activation = nn.Softmax(dim=1)

        elif activation == 'sigmoid':

            self.activation = nn.Sigmoid()

        else:

            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')



    def forward(self, x):

        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""

        x = self.encoder(x)

        x = self.decoder(x)

        return x



    def predict(self, x):



        if self.training:

            self.eval()



        with torch.no_grad():

            x = self.forward(x)

            if self.activation:

                x = self.activation(x)



        return x





class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_batchnorm=True):

        super().__init__()

        self.block = nn.Sequential(

            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),

            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),

        )



    def forward(self, x):

        x, skip = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')

        if skip is not None:

            x = torch.cat([x, skip], dim=1)

        x = self.block(x)

        return x





class CenterBlock(DecoderBlock):



    def forward(self, x):

        return self.block(x)





class UnetDecoder(Model):



    def __init__(

            self,

            encoder_channels,

            decoder_channels=(256, 128, 64, 32, 16),

            final_channels=1,

            use_batchnorm=True,

            center=False,

    ):

        super().__init__()



        if center:

            channels = encoder_channels[0]

            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)

        else:

            self.center = None



        in_channels = self.compute_channels(encoder_channels, decoder_channels)

        out_channels = decoder_channels



        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm)

        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm)

        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm)

        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm)

        self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm)

        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))



        self.initialize()



    def compute_channels(self, encoder_channels, decoder_channels):

        channels = [

            encoder_channels[0] + encoder_channels[1],

            encoder_channels[2] + decoder_channels[0],

            encoder_channels[3] + decoder_channels[1],

            encoder_channels[4] + decoder_channels[2],

            0 + decoder_channels[3],

        ]

        return channels



    def forward(self, x):

        encoder_head = x[0]

        skips = x[1:]



        if self.center:

            encoder_head = self.center(encoder_head)



        x = self.layer1([encoder_head, skips[0]])

        x = self.layer2([x, skips[1]])

        x = self.layer3([x, skips[2]])

        x = self.layer4([x, skips[3]])

        x = self.layer5([x, None])

        x = self.final_conv(x)



        return x





class ResNetEncoder(ResNet):



    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.pretrained = False

        del self.fc



    def forward(self, x):

        x0 = self.conv1(x)

        x0 = self.bn1(x0)

        x0 = self.relu(x0)



        x1 = self.maxpool(x0)

        x1 = self.layer1(x1)



        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        x4 = self.layer4(x3)



        return [x4, x3, x2, x1, x0]



    def load_state_dict(self, state_dict, **kwargs):

        state_dict.pop('fc.bias')

        state_dict.pop('fc.weight')

        super().load_state_dict(state_dict, **kwargs)





resnet_encoders = {

    'resnet18': {

        'encoder': ResNetEncoder,

#         'pretrained_settings': pretrained_settings['resnet18'],

        'out_shapes': (512, 256, 128, 64, 64),

        'params': {

            'block': BasicBlock,

            'layers': [2, 2, 2, 2],

        },

    },



    'resnet34': {

        'encoder': ResNetEncoder,

#         'pretrained_settings': pretrained_settings['resnet34'],

        'out_shapes': (512, 256, 128, 64, 64),

        'params': {

            'block': BasicBlock,

            'layers': [3, 4, 6, 3],

        },

    },



    'resnet50': {

        'encoder': ResNetEncoder,

#         'pretrained_settings': pretrained_settings['resnet50'],

        'out_shapes': (2048, 1024, 512, 256, 64),

        'params': {

            'block': Bottleneck,

            'layers': [3, 4, 6, 3],

        },

    },



    'resnet101': {

        'encoder': ResNetEncoder,

#         'pretrained_settings': pretrained_settings['resnet101'],

        'out_shapes': (2048, 1024, 512, 256, 64),

        'params': {

            'block': Bottleneck,

            'layers': [3, 4, 23, 3],

        },

    },



    'resnet152': {

        'encoder': ResNetEncoder,

#         'pretrained_settings': pretrained_settings['resnet152'],

        'out_shapes': (2048, 1024, 512, 256, 64),

        'params': {

            'block': Bottleneck,

            'layers': [3, 8, 36, 3],

        },

    },

}



encoders = {}

encoders.update(resnet_encoders)



def get_encoder(name, encoder_weights=None):

    Encoder = encoders[name]['encoder']

    encoder = Encoder(**encoders[name]['params'])

    encoder.out_shapes = encoders[name]['out_shapes']



    if encoder_weights is not None:

        pass

#         settings = encoders[name]['pretrained_settings'][encoder_weights]

#         encoder.load_state_dict(model_zoo.load_url(settings['url']))



    return encoder





def get_encoder_names():

    return list(encoders.keys())





def get_preprocessing_fn(encoder_name, pretrained='imagenet'):

    settings = encoders[encoder_name]['pretrained_settings']



    if pretrained not in settings.keys():

        raise ValueError('Avaliable pretrained options {}'.format(settings.keys()))



    input_space = settings[pretrained].get('input_space')

    input_range = settings[pretrained].get('input_range')

    mean = settings[pretrained].get('mean')

    std = settings[pretrained].get('std')

    

    return functools.partial(preprocess_input, mean=mean, std=std, input_space=input_space, input_range=input_range)





class Unet(EncoderDecoder):



    def __init__(

            self,

            encoder_name='resnet34',

            encoder_weights='imagenet',

            decoder_use_batchnorm=True,

            decoder_channels=(256, 128, 64, 32, 16),

            classes=1,

            activation='sigmoid',

            center=False,  # usefull for VGG models

    ):

        encoder = get_encoder(

            encoder_name,

            encoder_weights=encoder_weights

        )



        decoder = UnetDecoder(

            encoder_channels=encoder.out_shapes,

            decoder_channels=decoder_channels,

            final_channels=classes,

            use_batchnorm=decoder_use_batchnorm,

            center=center,

        )



        super().__init__(encoder, decoder, activation)



        self.name = 'u-{}'.format(encoder_name)



def mask2rle(img):



    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def test_trainforms():

    list_transforms = []

    list_transforms.extend(

        [

            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),

            ToTensor(),

        ]

    )

    list_trfms = Compose(list_transforms)

    return list_trfms



class TestDataset(Dataset):

    def __init__(self, root, df, mean, std):

        self.root = root

        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

        self.fnames = df['ImageId'].unique().tolist()

        self.num_samples = len(self.fnames)

        self.transform = Compose(

            [

                Normalize(mean=mean, std=std, p=1),

                ToTensor(),

            ]

        )



    def __getitem__(self, idx):

        fname = self.fnames[idx]

        path = os.path.join(self.root, fname)

        image = cv2.imread(path)

        images = self.transform(image=image)["image"]

        return fname, images



    def __len__(self):

        return self.num_samples



device = "cuda"

model = Unet("resnet18", encoder_weights="", classes=4, activation=None).to(device)



root = "/media/vuquan/DATA/data/kaggle/"

folder_test = "test_images"







sample_submission_path = '../input/severstal-steel-defect-detection/sample_submission.csv'

test_data_folder = "../input/severstal-steel-defect-detection/test_images"



ckpt_path = "../input/weight/resnet18.ptbest"

checkpoint = torch.load(ckpt_path)

model.load_state_dict(checkpoint['state_dict'])



# folder_test = root+folder_test

# list_file_test = glob.glob(folder_test)



# print(list_file_test)

# tmp = torch.zeros([1,3,256,1600]).to(device)

best_threshold = 0.5

num_workers = 2

batch_size = 10

print('best_threshold', best_threshold)



min_size = 3500

mean = (0.485, 0.456, 0.406)

std = (0.229, 0.224, 0.225)

df = pd.read_csv(sample_submission_path)

testset = DataLoader(

    TestDataset(test_data_folder, df, mean, std),

    batch_size=batch_size,

    shuffle=False,

    num_workers=num_workers,

    pin_memory=True

)



predictions = []

from tqdm import tqdm



for i, batch in enumerate(tqdm(testset)):

    fnames, images = batch

    batch_preds = torch.sigmoid(model(images.to(device)))

    batch_preds = batch_preds.detach().cpu().numpy()

    for fname, preds in zip(fnames, batch_preds):

        for cls, pred in enumerate(preds):

            pred, num = post_process(pred, best_threshold, min_size)

            rle = mask2rle(pred)

            name = fname + f"_{cls+1}"

            predictions.append([name, rle])



# save predictions to submission.csv

df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])

df.to_csv("submission.csv", index=False)



print(">>>>>> go go go <<<<<<<<")
!pip install ipdb