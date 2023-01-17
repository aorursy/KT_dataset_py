# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
os.listdir("../input/chest_xray/chest_xray/train/")
normal_images = os.listdir("../input/chest_xray/chest_xray/train/NORMAL/")

pnemonia_images = os.listdir("../input/chest_xray/chest_xray/train/PNEUMONIA/")

normal_images.remove(".DS_Store")

pnemonia_images.remove(".DS_Store")

train_normal, test_normal = train_test_split(normal_images, test_size = 0.1)

train_defect, test_defect = train_test_split(pnemonia_images, test_size = 0.1)
norm = cv2.imread("../input/chest_xray/chest_xray/train/NORMAL/" + normal_images[0], 0)

defect = cv2.imread("../input/chest_xray/chest_xray/train/PNEUMONIA/" + pnemonia_images[0], 0)

plt.figure(figsize = (9, 3))



plt.subplot(121)

plt.title("Normal")

plt.imshow(norm, cmap = 'gray')

plt.subplot(122)

plt.title("Pneumonia")

plt.imshow(defect, cmap = 'gray')
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



import random
class MyDataset(Dataset):

    def __init__(self, image_names, root_dir, label = 1, transforms = None):

        self.image_names = image_names

        self.root_dir = root_dir

        self.transform = transforms

        self.label = label

        

    def __len__(self):

        return len(self.image_names)

    

    def __getitem__(self, idx):

        img = cv2.imread(self.root_dir + self.image_names[idx])

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0

        

        return (img, self.label)

train_normal_data = MyDataset(image_names=train_normal, root_dir="../input/chest_xray/chest_xray/train/NORMAL/")

train_defect_data = MyDataset(image_names=train_defect, label=0, root_dir="../input/chest_xray/chest_xray/train/PNEUMONIA/")



test_normal_data = MyDataset(image_names=test_normal, root_dir="../input/chest_xray/chest_xray/train/NORMAL/")

test_defect_data = MyDataset(image_names=test_defect, label=0, root_dir="../input/chest_xray/chest_xray/train/PNEUMONIA/")
plt.subplot(121)

plt.title("Normal")

plt.imshow(train_normal_data[0][0][:, :, 0], cmap = 'gray')

plt.subplot(122)

plt.title("Normal")

plt.imshow(train_normal_data[1][0][:, :, 0], cmap = 'gray')
nloader = DataLoader(train_normal_data, batch_size=8)

ploader = DataLoader(train_defect_data, batch_size=8)



test_nloader = DataLoader(train_normal_data, batch_size=8)

test_ploader = DataLoader(train_defect_data, batch_size=8)
a = next(iter(nloader))

b = next(iter(ploader))



c = next(iter(test_nloader))

d = next(iter(test_ploader))
def create_batch(normal, defect):

    x = np.concatenate((normal[0].numpy(), defect[0].numpy()))

    y = np.concatenate((normal[1].numpy(), defect[1].numpy()))

    c = list(zip(x, y))

    random.shuffle(c)

    xx, yy = zip(*c)

    xx = np.stack(xx)

    yy = np.stack(yy)

    yy = yy.reshape(-1, 1)

    

    return xx, yy
x, y = create_batch(a, b)
from keras.applications.vgg16 import VGG16

from keras.layers import Input

	

model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(1024, 1024, 3)))
model.summary()
from keras.layers import Dense, Flatten, Dropout, AveragePooling2D

from keras.models import Model



trainable_model = model.output

trainable_model = AveragePooling2D(pool_size = (4, 4)) (trainable_model)

trainable_model = Flatten() (trainable_model)

trainable_model = Dense(128, activation='relu') (trainable_model)

trainable_model = Dropout(0.5) (trainable_model)

trainable_model = Dense(1, activation='sigmoid') (trainable_model)
m = Model(inputs=model.input, outputs = trainable_model)



for layer in model.layers:

    layer.trainable = False
m.summary()
from keras.optimizers import Adam



opt = Adam(lr=1e-04)



m.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['acc'])
from tqdm import tqdm_notebook as tq

with tq(total = 3) as jbar:

    for j in range(3):

        jbar.set_description("Epoch : {}".format(j+1))

        losses = 0

        accuracy = 0

        with tq(total = 30) as ibar:

            for i in range(30):

                a = next(iter(nloader))

                b = next(iter(ploader))

                x, y = create_batch(a, b)

                loss, acc = m.train_on_batch(x, y)

                losses += loss

                accuracy += acc

                ibar.set_description("Loss:{:.5f}, Acc:{:.5f}".format(losses/(i+1), accuracy/(i+1)))

                ibar.update(1)



        with tq(total = 8) as kbar:

            tloss = 0

            tacc = 0

            for k in range(8):

                c = next(iter(test_nloader))

                d = next(iter(test_ploader))

                x, y = create_batch(c, d)

                loss, acc = m.test_on_batch(x, y)

                tloss += loss

                tacc += acc

                kbar.set_description("Loss:{:.5f}, Acc:{:.5f}".format(tloss/(k+1), tacc/(k+1)))

                kbar.update(1)

        jbar.update(1)
normal_images = os.listdir("../input/chest_xray/chest_xray/test/NORMAL/")

pnemonia_images = os.listdir("../input/chest_xray/chest_xray/test/PNEUMONIA/")
train_normal_data = MyDataset(image_names=normal_images, root_dir="../input/chest_xray/chest_xray/test/NORMAL/")

train_defect_data = MyDataset(image_names=pnemonia_images, label=0, root_dir="../input/chest_xray/chest_xray/test/PNEUMONIA/")
nloader = DataLoader(train_normal_data, batch_size=8)

ploader = DataLoader(train_defect_data, batch_size=8)
with tq(total = 20) as kbar:

    tloss = 0

    tacc = 0

    for k in range(20):

        c = next(iter(nloader))

        d = next(iter(ploader))

        x, y = create_batch(c, d)

        loss, acc = m.test_on_batch(x, y)

        tloss += loss

        tacc += acc

        kbar.set_description("Loss:{:.5f}, Acc: {:.5f}".format(tloss/(k+1), tacc/(k+1)))

        kbar.update(1)
m.save("model.h5")
os.listdir(".")