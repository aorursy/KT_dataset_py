from matplotlib.pyplot import imshow

%matplotlib inline

import numpy as np

import torch

from torch import nn

import altair

from torchvision.transforms import transforms

from torchvision.transforms.functional import to_pil_image

import pandas as pd

from skimage import io, transform

from torch.utils.data import Dataset, DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train_labels = df_train.iloc[:, 0]

train_images = df_train.iloc[:, 1:]



print(train_images.shape)

transform = transforms.Compose([

    transforms.ToPILImage(), # but we don't need an image...

    transforms.ToTensor()

])
class MNISTDataset(Dataset):

    def __init__(self, images, labels=None, transforms=None):

        self.x = images

        self.y = labels

        self.transforms = transforms

        

    def __len__(self):

        return len(self.x)

    

    def __getitem__(self, idx):

        data = self.x.iloc[idx, :]

        data = np.asarray(data).astype(np.uint8).reshape(28, 28, 1) # ToPILImage transoform acts on

        # tensor of shape C x H x W or a numpy ndarray of shape H x W x C. I want the latter case.

        

        if self.transforms is not None:

            data = self.transforms(data)

        if self.y is not None:

            return data, self.y[idx]

        else:

            return data

train_data = MNISTDataset(train_images, train_labels, transform)

print(f'Initial train dataset: {len(train_data)}')



train_data, test_data = torch.utils.data.random_split(train_data, lengths=[37000, 5000])

print(f'New train dataset: {len(train_data)}, Test dataset: {len(test_data)}')



train_loader = DataLoader(train_data, batch_size=512, shuffle=True)

test_loader = DataLoader(test_data, batch_size=1028, shuffle=True)
class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=4)

        self.c2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4)

        self.pool = nn.MaxPool2d(kernel_size=4)

        self.flat = nn.Flatten()

        self.linear = nn.Linear(in_features=400, out_features=10)

        

    def forward(self, inputs):

        o1 = self.c1(inputs)

        o2 = self.c2(o1)

        o3 = self.pool(o2)

        o4 = self.flat(o3)

        o5 = self.linear(o4)

        return o5

        

n = Net().to(device)

# for i, _ in train_loader:

#     i = i.to(device)

#     print(n(i).shape)

#     break

loss_fn = nn.CrossEntropyLoss().to(device)

optim = torch.optim.Adam(n.parameters())
def train_step():    

    n.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)

        outputs = n(images).to(device)

        

        loss = loss_fn(outputs, labels)

        loss.backward()

        optim.step()

        optim.zero_grad()

        

    

    n.eval()

    

    with torch.no_grad():

        test_loss = sum([loss_fn(n(images.to(device)).to(device), labels.to(device)) for images, labels in test_loader])

        print(f'Test loss: {test_loss}')

        return test_loss

        

for i in range(3):

    train_step()
df_predict =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

predict_images  = df_predict.iloc[:, :]





print(predict_images.shape)



predict_data  = MNISTDataset(predict_images, transforms=transform)

predict_loader  = DataLoader(predict_data,  batch_size=1028, shuffle=False)
def predict():

    predictions = []

    n.eval()

    with torch.no_grad():

        for images in predict_loader:

            images = images.to(device)

            outputs = torch.nn.functional.softmax(n(images), dim=1)

            outputs = torch.argmax(outputs, dim=1).cpu().numpy()

            

            predictions.extend(list(outputs))

            

#             torch.argmax(

#                 ).cpu().numpy()

#             predictions.extend(outputs)

#             print(outputs)

    return predictions

predictions = predict()
with open('results.csv', 'w') as f:

    f.write('ImageId,Label\n')

    for item, pred in enumerate(predictions):

        f.write(f'{item+1},{pred}\n')