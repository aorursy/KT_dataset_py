import numpy as np 

import pandas as pd 

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

import torchvision.transforms as tfms

import torchvision.models as models



from torch.utils.data import DataLoader, Dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'



if device == 'cuda':

    torch.cuda.set_device(0)

    print(torch.cuda.current_device())
class CharData(Dataset):

    

    def __init__(self,

                 images,

                 labels=None,

                 transform=None,

                ):

        self.X = images

        self.y = labels

        

        self.transform = transform

    

    def __len__(self):

        return len(self.X)



    def __getitem__(self, idx):

        img = np.array(self.X.iloc[idx, :], dtype='uint8').reshape([28, 28, 1])

        if self.transform is not None:

            img = self.transform(img)

        

        if self.y is not None:

            y = np.zeros(10, dtype='float32')

            y[self.y.iloc[idx]] = 1

            return img, y

        else:

            return img
train_transform = tfms.Compose([

    tfms.ToPILImage(),

    tfms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),

    tfms.ToTensor()

])



# Test data without augmentation

test_transform = tfms.Compose([

    tfms.ToPILImage(),

    tfms.ToTensor()

])
df_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

target = df_train['label']

df_train.drop('label', axis=1, inplace=True)



X_test = pd.read_csv('../input/Kannada-MNIST/test.csv')

X_test.drop('id', axis=1, inplace=True)
X_train, X_dev, y_train, y_dev = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.3)

print('X_train', len(X_train))

print('X_dev', len(X_dev))

print('X_test', len(X_test))
train_dataset = CharData(X_train, y_train, train_transform)

dev_dataset = CharData(X_dev, y_dev, test_transform)

test_dataset = CharData(X_test, transform=test_transform)
BATCH_SIZE = 256



train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
fig, ax = plt.subplots(nrows=1, ncols=16, figsize=(30,4))



for batch in train_loader:

    for i in range(16):

        ax[i].set_title(batch[1][i].data.numpy().argmax())

        ax[i].imshow(batch[0][i, 0], cmap='gray')

    break
model = models.resnet50(pretrained=False)

model
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(2048, 10, bias=True)
model.to(device)
# class CustomModel(nn.Module):

#     def __init__(self):

#         super(CustomModel, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3)

#         self.fc1 = nn.Linear(24*24*32, 128)

#         self.fc2 = nn.Linear(128, 10)

        

#     def forward(self, x):

#         x = F.relu(self.conv1(x))

#         x = F.relu(self.conv2(x))

#         x = torch.flatten(x, start_dim=1)

#         x = F.relu(self.fc1(x))

#         return self.fc2(x)
# model = CustomModel().to(device)
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss()
def train_step(inputs, targets):

    optimizer.zero_grad()

    

    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)

    loss = criterion(outputs, targets.long().argmax(1))

    loss.backward()

    optimizer.step()

    

    batch_loss = loss.item()

    _, predictions = outputs.max(1)  # return values, indices

    correct = predictions.eq(targets.long().argmax(1)).sum().item()

    

    return batch_loss, correct





def test_step(inputs, targets):

    with torch.no_grad():

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets.long().argmax(1))

        

        batch_loss = loss.item()

        _, predictions = outputs.max(1)

        correct = predictions.eq(targets.long().argmax(1)).sum().item()

        

        return batch_loss, correct
EPOCHS = 10

best_acc = 0.



for epoch in range(EPOCHS):

    

    train_loss = 0.

    train_total = 0

    train_correct = 0

    test_loss = 0.

    test_total = 0

    test_correct = 0

    

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        batch_loss, correct = train_step(inputs, targets)

        train_loss += batch_loss

        train_total += targets.size(0)

        train_correct += correct

        

    for batch_idx, (inputs, targets) in enumerate(dev_loader):

        batch_loss, correct = test_step(inputs, targets)

        test_loss += batch_loss

        test_total += targets.size(0)

        test_correct += correct

        

    test_accuracy = (test_correct/test_total)*100

        

    template = 'Epoch {}, Loss: {:.6f}, Accuracy: {:.3f}%, Test Loss: {:.6f}, Test Accuracy: {:.3f}%'

    print(template.format(epoch+1,

                          train_loss,

                          (train_correct/train_total)*100,

                          test_loss,

                          test_accuracy))

    

    if test_accuracy > best_acc:

        print("new best acc!")

        best_acc = test_accuracy
model.eval()

predictions = []



for data in (test_loader):

    data = data.to(device)

    output = model(data).max(dim=1)[1] # argmax

    predictions += list(output.data.cpu().numpy())
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv('submission.csv', index=False)

submission.head()