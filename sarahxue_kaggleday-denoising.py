# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import time

from IPython.display import clear_output



import numpy as np

import torch

import torch.nn as nn



from matplotlib.pyplot import imshow, show, figure

from PIL import Image

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import RandomCrop, ToTensor, Compose, RandomHorizontalFlip, RandomVerticalFlip, ToPILImage
NUM_EPOCHS = 100

BATCH_SIZE = 32



PATCH_SIZE = 40

NOISE_SIGMA = 25

CROPS_PER_IMAGE = 10



TRAIN_DATA_PATH = "/kaggle/input/kaggle-days-denoise/data/train/"

VALIDATION_DATA_PATH = "/kaggle/input/kaggle-days-denoise/data/val/"
class DenoisingDataset(Dataset):

    def __init__(self, data_path):

        self._images = [np.array(Image.open(os.path.join(data_path, filename))) for filename in os.listdir(data_path)]

        self._sigma = NOISE_SIGMA

        self._transforms = Compose([

            ToPILImage(),

            RandomCrop(PATCH_SIZE),

            RandomHorizontalFlip(),

            RandomVerticalFlip(),

            ToTensor()])



    def __len__(self):

        return len(self._images) * CROPS_PER_IMAGE



    def __getitem__(self, item):

        image = self._images[item % len(self._images)]

        

        image = self._transforms(image)

        noise = torch.randn(image.size()).mul_(self._sigma / 255)

        

        noisy_image = (image + noise).clamp(0, 1)

        

        return noisy_image, image
class DnCNN(nn.Module):

    def __init__(self, depth=7, n_channels=16, image_channels=3, use_bnorm=True, kernel_size=3):

        super(DnCNN, self).__init__()

        kernel_size = 3

        padding = 1

        layers = []



        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))

        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth-2):

            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))

            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))

            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

        self._initialize_weights()



    def forward(self, x):

        out = self.dncnn(x)

        return x - out



    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                nn.init.orthogonal_(m.weight)

                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)
def visualize_validation(model):

    model.eval()

    image = Image.open(os.path.join(VALIDATION_DATA_PATH, "pier.png"))

    image = np.array(image).astype("float32") / 255.

    model_input = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)



    with torch.no_grad():

        result = model(model_input)



    result_image = result[0].clamp(0, 1).permute(1, 2, 0).numpy()

    result_image = (result_image * 255).astype("uint8")

    

    stacked_images = np.zeros((image.shape[0], image.shape[1] * 2, image.shape[2]), dtype="uint8")

    stacked_images[:, :image.shape[1]] = (image * 255).astype("uint8")

    stacked_images[:, image.shape[1]:] = result_image

    

    

    clear_output(wait=True)

    figure(figsize=(18, 18))

    imshow(stacked_images)

    show()

model = DnCNN()

dataset = DenoisingDataset(TRAIN_DATA_PATH)

criterion = nn.MSELoss(reduction="sum")

optimizer = Adam(model.parameters(), lr=0.001)
for epoch_id in range(NUM_EPOCHS):

    # First, we will use current model to predict a validation image

    visualize_validation(model)



    # Let's optimize parameters of our model

    model.train()

    

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)



    for iter_id, (input_images, target_images) in enumerate(data_loader):

        predicted_images = model(input_images)

        loss = criterion(predicted_images, target_images)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        print("\rEpoch {} Iteration {} Loss {}".format(epoch_id, iter_id, loss.item() / BATCH_SIZE), end="")