# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os

import random

import cv2

import time

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torchvision.transforms as transforms

from collections import OrderedDict

from torchvision.models import resnet18, resnet34

from torch.utils.data import Dataset, DataLoader



os.chdir('/kaggle/input/pretrainedmodels/models')

from senet import *

os.chdir('/kaggle/working/')





########################################################################################################################

def get_model(config, num_classes=196):

    model_name = config.MODEL

    f = globals().get(model_name)



    if model_name.startswith('resnet'):

        model = f(pretrained=False)

        model.avgpool = nn.AdaptiveAvgPool2d(1)

        in_features = model.fc.in_features

        model.fc = nn.Linear(in_features, num_classes)

    else:

        model = f(num_classes=1000, pretrained=None)

        model.avg_pool = nn.AdaptiveAvgPool2d(1)

        in_features = model.last_linear.in_features

        model.last_linear = nn.Linear(in_features, num_classes)



    print('model name:', model_name)

    return model





class CarDataset(Dataset):

    def __init__(self, config, transform=None):

        self.config = config

        self.transform = transform

        self.frame = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/test.csv')



    def __len__(self):

        return self.frame.shape[0]



    def __getitem__(self, idx):

        image = cv2.imread(os.path.join('/kaggle/input/2019-3rd-ml-month-with-kakr/test', self.frame["img_file"][idx]), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        x1, y1, x2, y2 = self.frame.iloc[idx][1:5]

        image = image[y1:y2 + 1, x1:x2 + 1]



        if image.shape[0] / image.shape[1] > 0.536:

            w_offset = int(((image.shape[0] / 0.536) - image.shape[1]) / 2)

            image = np.pad(image, ((0, 0), (w_offset, w_offset), (0, 0)), 'constant', constant_values=(0))

        else:

            h_offset = int(((image.shape[1] * 0.536) - image.shape[0]) / 2)

            image = np.pad(image, ((h_offset, h_offset), (0, 0), (0, 0)), 'constant', constant_values=(0))



        image = cv2.resize(image, (self.config.IMG_W, self.config.IMG_H))



        if self.transform is not None:

            image = self.transform(image)



        return image





def get_dataloader(config, transform=None):

    dataset = CarDataset(config, transform)



    dataloader = DataLoader(dataset,

                             shuffle=False,

                             batch_size=config.BATCH_SIZE,

                             num_workers=config.NUM_WORKERS,

                             pin_memory=True)



    return dataloader





class Normalize:

    """

    normalize data to -1 ~ 1. channelwise! (c, h, w)

    """

    def __call__(self, data):

        smooth = 1e-6

        # for i in range(data.shape[0]):

        #     data[i] = (data[i] - np.min(data[i]) + smooth) / (np.max(data[i]) - np.min(data[i]) + smooth)



        data = (data - np.min(data) + smooth) / (np.max(data) - np.min(data) + smooth)

        data = data * 2 - 1



        return data





class ToTensor:

    """

    convert ndarrays to Tensors.

    """

    def __call__(self, data):

        data = np.transpose(data, (2, 0, 1))

        data = torch.from_numpy(data).float()

        return data





class HorizontalFlip:

    def __call__(self, data):

        return data[:,::-1]



########################################################################################################################





def inference(model, dataloader):

    model.eval()



    output = []

    with torch.no_grad():

        start = time.time()

        for i, images in enumerate(dataloader):

            images = images.cuda()

            logits = model(images)



            preds = logits.detach().cpu().numpy()

            # print(preds.shape)



            output.append(preds)



            del images, logits, preds

            torch.cuda.empty_cache()



            end = time.time()

            print('[%2d/%2d] time: %.2f' % (i, len(dataloader), end - start))



    output = np.concatenate(tuple(output), axis=0)

    return output





def run(config):

    model = get_model(config).cuda()



    checkpoint = torch.load(config.CHECKPOINT)



    state_dict_old = checkpoint['state_dict']

    state_dict = OrderedDict()

    # delete 'module.' because it is saved from DataParallel module

    for key in state_dict_old.keys():

        if key.startswith('module.'):

            state_dict[key[7:]] = state_dict_old[key]

        else:

            state_dict[key] = state_dict_old[key]



    model.load_state_dict(state_dict)



    # TTA

    test_loader = get_dataloader(config, transform=transforms.Compose([Normalize(),

                                                                       ToTensor()]))

    out = inference(model, test_loader)



    test_loader_flip = get_dataloader(config, transform=transforms.Compose([HorizontalFlip(),

                                                                            Normalize(),

                                                                            ToTensor()]))

    out_flip = inference(model, test_loader_flip)



    output = out + out_flip

    print('tta flip inference finished. shape:', output.shape)



    return output





def seed_everything():

    seed = 2019

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True





def main():

    import warnings

    warnings.filterwarnings("ignore")

    seed_everything()



    config_0 = Config(checkpoint='/kaggle/input/5foldresnext/epoch_0032_0.9493.pth')

    fold_0 = run(config_0)



    config_1 = Config(checkpoint='/kaggle/input/5foldresnext/epoch_0033_0.9435.pth')

    fold_1 = run(config_1)



    config_2 = Config(checkpoint='/kaggle/input/5foldresnext/epoch_0049_0.9309.pth')

    fold_2 = run(config_2)



    config_3 = Config(checkpoint='/kaggle/input/5foldresnext/epoch_0053_0.9428.pth')

    fold_3 = run(config_3)



    config_4 = Config(checkpoint='/kaggle/input/5foldresnext/epoch_0054_0.9437.pth')

    fold_4 = run(config_4)



    final = fold_0 + fold_1 + fold_2 + fold_3 + fold_4

    final = np.argmax(final, axis=1) + 1



    submission = pd.read_csv('/kaggle/input/2019-3rd-ml-month-with-kakr/sample_submission.csv')

    submission['class'] = final

    submission.to_csv('submission.csv', index=False)



    print('success!')





class Config():

    def __init__(self, checkpoint='/kaggle/input/5fold-resnext/epoch_0032_0.9493.pth', model='se_resnext50_32x4d'):

        self.CHECKPOINT = checkpoint

        self.MODEL = model

        self.IMG_H = 256

        self.IMG_W = 480

        self.BATCH_SIZE = 128

        self.NUM_WORKERS = 2





if __name__ == '__main__':

    main()
