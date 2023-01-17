### Transfer Learning



import numpy as np

import pandas as pd

import os

from tqdm import tqdm

import matplotlib.image as mpimg

import matplotlib.pyplot as plt



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torchvision.models as models

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms



plt.ion()   # interactive mode
import pandas as pd



df = pd.read_csv('../input/train.csv')

print(len(df))

df.head()
train_dir = '../input/train/train/'

test_dir = '../input/test/test/'
class ImageData(Dataset):

    def __init__(self, df, data_dir, transform):

        super().__init__()

        self.df = df

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):       

        img_name = self.df.id[index]

        label = self.df.has_cactus[index]

        

        img_path = os.path.join(self.data_dir, img_name)

        image = mpimg.imread(img_path)

        image = self.transform(image)

        return image, label
epochs = 15

batch_size = 20

device = torch.device('cuda:0')
data_transf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

train_data = ImageData(df = df, data_dir = train_dir, transform = data_transf)

train_loader = DataLoader(dataset = train_data, batch_size = batch_size)
# train_loader = DataLoader(

#             ImageFilelist(root="../input/train/train", flist=train,

#              transform=transforms.Compose([transforms.RandomSizedCrop(224),

#                  transforms.RandomHorizontalFlip(),

#                  transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#             ])),

#             batch_size=64, shuffle=False,

#             num_workers=4, pin_memory=True)



# val_loader = torch.utils.data.DataLoader(

#             ImageFilelist(root="../input/train/train", flist=train,

#              transform=transforms.Compose([transforms.Scale(256),

#                  transforms.CenterCrop(224),

#                  transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#             ])),

#             batch_size=16, shuffle=False,

#             num_workers=1, pin_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)

model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()
model
%%time

# Train model

for epoch in tqdm(range(epochs)):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)

        labels = labels.to(device)

        

        # Forward

        outputs = model(images)

        loss = loss_func(outputs, labels)

        

        # Backward and optimize

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        

        if (i+1) % 500 == 0:

            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
submit = pd.read_csv('../input/sample_submission.csv')

test_data = ImageData(df = submit, data_dir = test_dir, transform = data_transf)

test_loader = DataLoader(dataset = test_data, shuffle=False)
predict = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.to(device), target.to(device)

    output = net(data)

    

    _, pred = torch.max(output.data, 1)

    predict.append(int(pred))

    

submit['has_cactus'] = predict

submit.to_csv('submission.csv', index=False)
submit.head()