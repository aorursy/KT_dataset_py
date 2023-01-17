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
import matplotlib.pyplot as plt
print(os.listdir('../input/generative-dog-images'))
!unzip ../input/generative-dog-images/all-dogs.zip -d /kaggle/working/ > /dev/null
PATH = '/kaggle/working/all-dogs/'

images = os.listdir(PATH)

def plotDog():

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))



    for indx, axis in enumerate(axes.flatten()):

        rnd_indx = np.random.randint(0, len(os.listdir(PATH)))

        img = plt.imread(PATH + images[rnd_indx])

        imgplot = axis.imshow(img)

        axis.set_title(images[rnd_indx])

        axis.set_axis_off()

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return plt.tight_layout(rect=[0, 0.03, 1, 0.95])



    
plotDog()
print(f'There are {len(images)} pictures of dogs.')



fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,10))



for indx, axis in enumerate(axes.flatten()):

    rnd_indx = np.random.randint(0, len(os.listdir(PATH)))

    img = plt.imread(PATH + images[rnd_indx])

    imgplot = axis.imshow(img)

    axis.set_title(images[rnd_indx])

    axis.set_axis_off()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

batch_size = 32

image_size = 64



random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]

transform = transforms.Compose([transforms.Resize(64),

                                transforms.CenterCrop(64),

                                transforms.RandomHorizontalFlip(p=0.5),

                                transforms.RandomApply(random_transforms, p=0.2),

                                transforms.ToTensor(),

                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



train_data = dset.ImageFolder('/kaggle/working/', transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

                                           

imgs, label = next(iter(train_loader))

imgs = imgs.numpy().transpose(0, 2, 3, 1)