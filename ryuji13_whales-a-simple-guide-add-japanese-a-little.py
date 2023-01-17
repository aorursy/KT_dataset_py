import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import cv2

import os

from PIL import Image



from matplotlib.pyplot import imshow

from IPython.display import HTML

print(os.listdir('../input'))
img_train_path = os.path.abspath('../input/train')

img_test_path = os.path.abspath('../input/test')

csv_train_path = os.path.abspath('../input/train.csv')

csv_train_path
df = pd.read_csv(csv_train_path)

df.head()
df['Image_path'] = [os.path.join(img_train_path,whale) for whale in df['Image']]
df.head()
full_path_random_whales = np.random.choice(df['Image_path'],5)
full_path_random_whales
%matplotlib inline

for whale in full_path_random_whales:

    img = Image.open(whale)

    plt.imshow(img)

    plt.show()
from torchvision import transforms
img = cv2.imread(full_path_random_whales[0])

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

res = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

plt.imshow(res,cmap='gray')

plt.show()
normalize = transforms.Normalize(

   mean=[0.485, 0.456, 0.406],

   std=[0.229, 0.224, 0.225]

)

preprocess = transforms.Compose([

   transforms.Grayscale(num_output_channels=1),

   transforms.Resize(128),

   transforms.CenterCrop(128),

   transforms.ToTensor(),

   normalize

])

imgs = [Image.open(whale) for whale in full_path_random_whales]

imgs_tensor = [preprocess(whale) for whale in imgs]
imgs_tensor[0].shape
img = imgs_tensor[0]

plt.imshow(img[0],cmap='gray')

plt.show()
df.Id.value_counts().head()
I_dont_want_new_whales = df['Id'] != 'new_whale'

df = df[I_dont_want_new_whales]

df.Id.value_counts().head()
unique_classes = pd.unique(df['Id'])

encoding = dict(enumerate(unique_classes))

encoding = {value: key for key, value in encoding.items()}

df = df.replace(encoding)
df.head()
import torch

import torch.nn.functional as F

from torchvision import datasets, transforms

from torch import nn

from torch import optim

from torch.utils.data import Dataset, DataLoader
test = df['Image_path'][:1000]

imgs = [Image.open(whale) for whale in test]

imgs_tensor = torch.stack([preprocess(whale) for whale in imgs])
labels = torch.tensor(df['Id'][:1000].values)

max_label = int(max(labels)) +1

max_label
plt.imshow(imgs_tensor[0].reshape(128,128),cmap='gray')
model = nn.Sequential(nn.Linear(128*128, 256),

                      nn.Sigmoid(),

                      nn.Linear(256, 128),

                      nn.Sigmoid(),

                      nn.Linear(128, max_label),

                      nn.LogSoftmax(dim=1))



optimizer = optim.SGD(model.parameters(), lr=0.01)

criterion = nn.NLLLoss()



model
epochs = 5

batch_size = 10

iters = int(len(imgs_tensor)/batch_size)

next_batch = 0

for e in range(epochs):

    running_loss = 0

    next_batch = 0

    for n in range(iters):

        batch_images = imgs_tensor[next_batch:next_batch+batch_size] 

        batch_images = batch_images.view(batch_images.shape[0], -1)

        batch_labels = labels[next_batch:next_batch+batch_size]

        

        optimizer.zero_grad()

        

        output = model(batch_images)

        loss = criterion(output, batch_labels)

        

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

        next_batch += batch_size

        

    print(running_loss)