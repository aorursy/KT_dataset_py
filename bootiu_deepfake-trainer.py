%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import gc

import cv2

import glob

import time

import copy

from tqdm import tqdm_notebook as tqdm

from PIL import Image

from sklearn.model_selection import train_test_split



import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torch.nn.functional as F

import torchvision

from torchvision import models, transforms
trained_weights_path = '../input/deepfake-model-weights/vgg19_ep5_20191219.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_dir = '../input/deepfake-detection-challenge/test_videos'

os.listdir(test_dir)[:5]
def get_img_from_mov(video_file, show_img=False):

    # https://note.nkmk.me/python-opencv-videocapture-file-camera/

    cap = cv2.VideoCapture(video_file)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    

    image_list = []

    for i in range(frames):

        _, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_list.append(image)

    cap.release()



    if show_img:

        fig, ax = plt.subplots(1,1, figsize=(15, 15))

        ax.imshow(image[0])

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        ax.title.set_text(f"FRAME 0: {video_file.split('/')[-1]}")

        plt.grid(False)

        

    return image_list



def detect_face(img):

    # Add Dataset "Haarcascades"

    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_alt.xml')

    face_crops = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    

    if len(face_crops) == 0:

        return []

    

    crop_imgs = []

    for i in range(len(face_crops)):

        x = face_crops[i][0]

        y = face_crops[i][1]

        w = face_crops[i][2]

        h = face_crops[i][3]

        #x,y,w,h=ratio*x,ratio*y,ratio*w,ratio*h

        crop_imgs.append(img[y:y+h, x:x+w])

    return crop_imgs
class Resize(object):

    def __init__(self, size=300):

        self.size = size



    def __call__(self, image):

        image = cv2.resize(image, (self.size, self.size))

        return image



# Data Augumentation

class ImageTransform():

    def __init__(self, resize):

        self.data_transform = {

            'test': transforms.Compose([

                Resize(resize),

                transforms.ToTensor(),

            ])

        }

        

    def __call__(self, img, phase):

        return self.data_transform[phase](img)





class DeepfakeDataset(Dataset):

    def __init__(self, file_list, transform=None, phase='test'):

        self.file_list = file_list

        self.transform = transform

        self.phase = phase

        

    def __len__(self):

        return len(self.file_list)

    

    def __getitem__(self, idx):

        

        mov_path = self.file_list[idx]

        # first frame image only

        image = get_img_from_mov(mov_path, show_img=False)[0]

        # FaceCrop

        image = detect_face(image)[0]

        # Transform

        image = self.transform(image, self.phase)

        

        return image, mov_path
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

__all__ = [

    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',

    'vgg19_bn', 'vgg19',

]





model_urls = {

    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',

    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',

    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',

    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',

    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',

    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',

    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',

    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',

}





class VGG(nn.Module):



    def __init__(self, features, num_classes=1000, init_weights=True):

        super(VGG, self).__init__()

        self.features = features

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(

            nn.Linear(512 * 7 * 7, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, 4096),

            nn.ReLU(True),

            nn.Dropout(),

            nn.Linear(4096, num_classes),

        )

        if init_weights:

            self._initialize_weights()



    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x



    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):

                nn.init.normal_(m.weight, 0, 0.01)

                nn.init.constant_(m.bias, 0)





def make_layers(cfg, batch_norm=False):

    layers = []

    in_channels = 3

    for v in cfg:

        if v == 'M':

            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:

                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            else:

                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v

    return nn.Sequential(*layers)





cfgs = {

    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],

    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}





def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):

    if pretrained:

        kwargs['init_weights'] = False

    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:

        state_dict = load_state_dict_from_url(model_urls[arch],

                                              progress=progress)

        model.load_state_dict(state_dict)

    return model





def vgg11(pretrained=False, progress=True, **kwargs):

    r"""VGG 11-layer model (configuration "A") from

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)





def vgg11_bn(pretrained=False, progress=True, **kwargs):

    r"""VGG 11-layer model (configuration "A") with batch normalization

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)





def vgg13(pretrained=False, progress=True, **kwargs):

    r"""VGG 13-layer model (configuration "B")

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)





def vgg13_bn(pretrained=False, progress=True, **kwargs):

    r"""VGG 13-layer model (configuration "B") with batch normalization

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)





def vgg16(pretrained=False, progress=True, **kwargs):

    r"""VGG 16-layer model (configuration "D")

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)





def vgg16_bn(pretrained=False, progress=True, **kwargs):

    r"""VGG 16-layer model (configuration "D") with batch normalization

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)





def vgg19(pretrained=False, progress=True, **kwargs):

    r"""VGG 19-layer model (configuration "E")

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)





def vgg19_bn(pretrained=False, progress=True, **kwargs):

    r"""VGG 19-layer model (configuration 'E') with batch normalization

    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:

        pretrained (bool): If True, returns a model pre-trained on ImageNet

        progress (bool): If True, displays a progress bar of the download to stderr

    """

    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
model = vgg19()

model.classifier[6] = nn.Linear(in_features=4096, out_features=2)

model.load_state_dict(torch.load(trained_weights_path))
print(model)
test_file = [os.path.join(test_dir, path) for path in os.listdir(test_dir)]
test_file[:5]
# Prediction

pred_list = []

path_list = []



with torch.no_grad():

    for mov_path in tqdm(test_file):

        pred = 0

        try:

            # first frame image only

            img = get_img_from_mov(mov_path, show_img=False)

            # All Frame Image

            # If face cannot be detected, prediction is 0.5

            _img = img[0]

            # FaceCrop

            _img = detect_face(_img)[0]

            # Transform

            _img = ImageTransform(resize=224)(_img, 'test')



            _img = _img.unsqueeze(0)

            _img = _img.to(device)

            model.to(device)

            model.eval()



            output = model(_img)

            pred += F.softmax(output, dim=1)[:, 1].tolist()[0]

            

            del img, _img

            gc.collect()

        except:

            pred += 0.5

        

        pred_list.append(pred)

        path_list.append(mov_path.split('/')[-1])

        

torch.cuda.empty_cache()
# Submission

res = pd.DataFrame({

    'filename': path_list,

    'label': pred_list,

})



res.sort_values(by='filename', ascending=True, inplace=True)



res.to_csv('submission.csv', index=False)