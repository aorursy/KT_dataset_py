!pip install /kaggle/input/testtimeaug/ttach-0.0.2-py3-none-any.whl
import os



import math

import openslide

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import albumentations

from tqdm import tqdm

from joblib import Parallel, delayed

from matplotlib import pyplot as plt

from PIL import Image, ImageChops



import cv2



import torch

import torch.utils.data as data_utils

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from torch.optim import lr_scheduler

from torch import nn



import ttach as tta

from torchvision import transforms,models

import torch.nn.functional as F

from tqdm.auto import tqdm

from torch import Tensor



from collections import OrderedDict
!ls /kaggle/input/prostate-cancer-grade-assessment
BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment'

# DATA_DIR = os.path.join(BASE_DIR, 'train_images')

DATA_DIR = os.path.join(BASE_DIR, 'test_images')

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# test_df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'))[['image_id', 'data_provider']].loc[:10]

# test_df
test_df = pd.read_csv(os.path.join(BASE_DIR, 'test.csv'))

sample_sub_df = pd.read_csv(os.path.join(BASE_DIR, 'sample_submission.csv'))
org_test_df = test_df.copy()

org_test_df
crop_size = 256  # Size of resultant images

crop_level = 2  # The level of slide used to get the images (you can use 0 to get very high resolution images)

down_samples = [1, 4, 16]  # List of down samples available in any tiff image file
def split_image(openslide_image):

    """

    Splits the given image into multiple images if 256x256

    """

    

    # Get the size of the given image

    width, height = openslide_image.level_dimensions[crop_level]



    # Get the dimensions of level 0 resolution, as it's required in "read_region()" function

    base_height = down_samples[crop_level] * height  # height of level 0

    base_width = down_samples[crop_level] * width  # width of level 0



    # Get the number of smaller images 

    h_crops = math.ceil(width / crop_size)

    v_crops = math.ceil(height / crop_size)



    splits = []

    for v in range(v_crops):

        for h in range(h_crops): 

            x_location = h*crop_size*down_samples[crop_level]

            y_location = v*crop_size*down_samples[crop_level]



            patch = openslide_image.read_region((x_location, y_location), crop_level, (crop_size, crop_size))



            splits.append(patch)

    return splits, h_crops, v_crops
def get_emptiness(arr):

    total_ele = arr.size

    white_ele = np.count_nonzero(arr == 255) + np.count_nonzero(arr == 0)

    return white_ele / total_ele
ignore_threshold = 0.95  # If the image is more than 95% empty, consider it as white and ignore
def filter_white_images(images):

    non_empty_crops = []

    for image in images:

        image_arr = np.array(image)[...,:3]  # Discard the alpha channel

        emptiness = get_emptiness(image_arr)

        if emptiness < ignore_threshold:

            non_empty_crops.append(image)

    return non_empty_crops
dataset = []

def create_dataset(count):

    img = os.path.join(DATA_DIR, f'{test_df["image_id"].iloc[count]}.tiff')

    img = openslide.OpenSlide(img)

    crops, _, _ = split_image(img)

    img.close()



    non_empty_crops = filter_white_images(crops)

    image_id = test_df['image_id'].iloc[count]



    for index, img in enumerate(non_empty_crops):

        img_metadata = {}

        img = img.convert('RGB')



        img_metadata['image_id'] = f'{image_id}_{index}'

        img_metadata['data_provider'] = test_df['data_provider'].iloc[count]

        img_metadata['group'] = count



        img.save(f'{image_id}_{index}.jpg', 'JPEG', quality=100, optimize=True, progressive=True)

        dataset.append(img_metadata)

    return dataset
if os.path.exists(DATA_DIR):

    dataset = Parallel(n_jobs=8)(delayed(create_dataset)(count) for count in tqdm(range(len(test_df))))

    dataset = [item for sublist in dataset for item in sublist]



    dataset = pd.DataFrame(dataset)

    dataset.to_csv('new_test.csv', index=False)

    test_df = pd.read_csv('new_test.csv')
class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):

        super(_Transition, self).__init__()

        self.add_module('norm', nn.BatchNorm2d(num_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,

                                          kernel_size=1, stride=1, bias=False))

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
class _DenseLayer(nn.Module):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):

        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *

                                           growth_rate, kernel_size=1, stride=1,

                                           bias=False)),

        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),

        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,

                                           kernel_size=3, stride=1, padding=1,

                                           bias=False)),

        self.drop_rate = float(drop_rate)

        self.memory_efficient = memory_efficient



    def bn_function(self, inputs):

        # type: (List[Tensor]) -> Tensor

        concated_features = torch.cat(inputs, 1)

        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484

        return bottleneck_output



    # todo: rewrite when torchscript supports any

    def any_requires_grad(self, input):

        # type: (List[Tensor]) -> bool

        for tensor in input:

            if tensor.requires_grad:

                return True

        return False



    @torch.jit.unused  # noqa: T484

    def call_checkpoint_bottleneck(self, input):

        # type: (List[Tensor]) -> Tensor

        def closure(*inputs):

            return self.bn_function(*inputs)



        return cp.checkpoint(closure, input)



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (List[Tensor]) -> (Tensor)

        pass



    @torch.jit._overload_method  # noqa: F811

    def forward(self, input):

        # type: (Tensor) -> (Tensor)

        pass



    # torchscript does not yet support *args, so we overload method

    # allowing it to take either a List[Tensor] or single Tensor

    def forward(self, input):  # noqa: F811

        if isinstance(input, Tensor):

            prev_features = [input]

        else:

            prev_features = input



        if self.memory_efficient and self.any_requires_grad(prev_features):

            if torch.jit.is_scripting():

                raise Exception("Memory Efficient not supported in JIT")



            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)

        else:

            bottleneck_output = self.bn_function(prev_features)



        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:

            new_features = F.dropout(new_features, p=self.drop_rate,

                                     training=self.training)

        return new_features
class _DenseBlock(nn.ModuleDict):

    _version = 2



    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):

        super(_DenseBlock, self).__init__()

        for i in range(num_layers):

            layer = _DenseLayer(

                num_input_features + i * growth_rate,

                growth_rate=growth_rate,

                bn_size=bn_size,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient,

            )

            self.add_module('denselayer%d' % (i + 1), layer)



    def forward(self, init_features):

        features = [init_features]

        for name, layer in self.items():

            new_features = layer(features)

            features.append(new_features)

        return torch.cat(features, 1)
class DenseNet(nn.Module):

    r"""Densenet-BC model class, based on

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:

        growth_rate (int) - how many filters to add each layer (`k` in paper)

        block_config (list of 4 ints) - how many layers in each pooling block

        num_init_features (int) - the number of filters to learn in the first convolution layer

        bn_size (int) - multiplicative factor for number of bottle neck layers

          (i.e. bn_size * k features in the bottleneck layer)

        drop_rate (float) - dropout rate after each dense layer

        num_classes (int) - number of classification classes

        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,

          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_

    """



    __constants__ = ['features']



    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),

                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):



        super(DenseNet, self).__init__()



        # First convolution

        self.features = nn.Sequential(OrderedDict([

            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # 3 is number of channels in input image

            ('norm0', nn.BatchNorm2d(num_init_features)),

            ('relu0', nn.ReLU(inplace=True)),

            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ]))



        # Each denseblock

        num_features = num_init_features

        for i, num_layers in enumerate(block_config):

            block = _DenseBlock(

                num_layers=num_layers,

                num_input_features=num_features,

                bn_size=bn_size,

                growth_rate=growth_rate,

                drop_rate=drop_rate,

                memory_efficient=memory_efficient

            )

            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:

                trans = _Transition(num_input_features=num_features,

                                    num_output_features=num_features // 2)

                self.features.add_module('transition%d' % (i + 1), trans)

                num_features = num_features // 2



        # Final batch norm

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))



        # Linear layer

        self.classifier = nn.Linear(num_features, num_classes)



        # Official init from torch repo.

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.constant_(m.bias, 0)



    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out
def _densenet(growth_rate, block_config, num_init_features, **kwargs):

    return DenseNet(growth_rate, block_config, num_init_features, **kwargs)
def densenet121(**kwargs):

    r"""Densenet-121 model from

    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    """

    return _densenet(32, (6, 12, 24, 16), 64, **kwargs)
class DenseNet121Wrapper(nn.Module):

    def __init__(self):

        super(DenseNet121Wrapper, self).__init__()

        

        # Load imagenet pre-trained model 

        self.dense_net = densenet121()

        

        # Appdend output layers based on our date

        self.out = nn.Linear(in_features=1000, out_features=6)

        

    def forward(self, X):

        output = self.dense_net(X)

        output = self.out(output)

        

        return output
paths = ['panda-densenet121-training-images-fold0',

        'panda-densenet121-training-images-fold1',

        'panda-densenet121-training-fold2',

        'panda-densenet121-training-fold3',

        'panda-densenet121-training-fold4']
FOLDS = 5
# Define TTA parameters

transforms = tta.Compose(

    [

        tta.HorizontalFlip(),

        tta.VerticalFlip(),

        tta.Rotate90([0, 90]),

        tta.Scale(scales=[1, 2]),

        tta.Multiply(factors=[0.9, 1, 1.1]),

    ]

)
models = []

for fold in range(FOLDS):

    model = DenseNet121Wrapper()

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(f'/kaggle/input/{paths[fold]}/densenet121_256_fold{fold}.pth', map_location=DEVICE))

    model = tta.ClassificationTTAWrapper(model, transforms)

    model.eval()

    models.append(model)
WORKING_DIR = os.path.join('/', 'kaggle', 'working')
class PandaDataset(Dataset):

    """Custom dataset for PANDA Tests"""

    

    def __init__(self, df, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.df = df

        self.aug = albumentations.Compose([

            albumentations.Normalize(mean, std, always_apply=True)

        ])

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        image_id = self.df.loc[index]['image_id']

        image = cv2.imread(os.path.join(WORKING_DIR, f'{image_id}.jpg'))

        image = self.aug(image=image)['image']

        

        # Convert from NHWC to NCHW as pytorch expects images in NCHW format

        image = np.transpose(image, (2, 0, 1))

        

        # For now, just return image and ISUP grades

        return image
BATCH_SIZE=16
def inference(model, test_loader, device):

    preds = []

    for i, images in tqdm(enumerate(test_loader)):

        images = images.to(device, dtype=torch.float)/255

            

        with torch.no_grad():

            y_preds = model(images)



#         preds.append(y_preds.to('cpu').numpy().argmax(1))

#         print(preds.append(y_preds.to('cpu').numpy().argmax(1)))

        preds.append(y_preds.to('cpu').numpy())

    preds = np.concatenate(preds)

    return preds
def submit(sample):

    global sample_sub_df

    if os.path.exists(DATA_DIR):

        test_dataset = PandaDataset(test_df)

        test_loader = data_utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        

        preds = []

        for fold in range(FOLDS):

            preds.append(inference(models[fold], test_loader, DEVICE))

        preds = np.array(preds)

        preds = np.argmax(np.mean(preds, axis=0), axis=1)

        

        test_df['preds'] = preds

        sample = sample.drop(['data_provider'], axis=1)

        sample['isup_grade'] = test_df.groupby('group')['preds'].agg(lambda x:x.value_counts().index[0])

        return sample

    return sample_sub_df
submission = submit(org_test_df)

submission['isup_grade'] = submission['isup_grade'].astype(int)

submission.head()
!rm -rf *
submission.to_csv('submission.csv', index=False)