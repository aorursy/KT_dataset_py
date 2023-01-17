# !pip install --upgrade wandb -q

# !wandb login <API KEY>
import torch

import torchvision.transforms as transforms

import pandas as pd

import os

import matplotlib.pyplot as plt

import wandb

# %matplotlib inline

# from PIL import Image
ds_dir = '/kaggle/input/sign-language-mnist/'



epochs = 25

lr = 0.001

# momentum = 0.9

train_batch_size = 12000 #27455

test_batch_size = 5000 #7172

wb = False
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):

        csv_dir = os.path.join(root_dir, csv_file)

        csv = pd.read_csv(csv_dir)

        self.labels = csv.iloc[:,0]

        self.images = csv.iloc[:,1:]

        self.images = (self.images.values).reshape(-1,28,28).astype('float32')

        self.labels = torch.tensor(self.labels)

#         self.labels = F.one_hot(self.labels)

        if transform:

            for i in range(self.images.shape[0]):

                self.images[i] = transform(self.images[i])

    

    def __len__(self):

        return len(self.labels)

    

    def __getitem__(self, index):

#         if torch.is_tensor(index):

#             index = index.tolist()

        

        img = self.images[index]

        label = self.labels[index]

    

        sample = {'image': img, 'label': label}

        

        return sample



# ds = dataset('sign_mnist_test/sign_mnist_test.csv', ds_dir, transform=None)

# ds[0]['label']
# csv = pd.read_csv(os.path.join(ds_dir, 'sign_mnist_test/sign_mnist_test.csv'))

# # print(csv.loc[csv['label'] == 9].head())

# # img = csv.iloc[21, 1:]

# # img = (img.values).reshape(28,28).astype('float32')

# # plt.imshow(img, 'gray')

# labels = F.one_hot(torch.tensor(csv.iloc[:,0]))

# for i in range (50):

#     print('eeky' if labels[i][9] != 0 else '', end='')
transform = transforms.Compose(

    [transforms.ToPILImage(),

     transforms.RandomRotation(degrees=5),

     transforms.ToTensor(),

     transforms.Normalize(0.5, 0.5)])



trainset = dataset('sign_mnist_train/sign_mnist_train.csv', ds_dir, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,

                                          shuffle=True, num_workers=1)



testset = dataset('sign_mnist_test/sign_mnist_test.csv', ds_dir, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,

                                          shuffle=True, num_workers=1)



classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'] # no 'J' or 'Z'
# import string

# for i in string.ascii_uppercase:

#     print(f'\'{i}\', ', end='')
def show_imgs(image, label):

    plt.imshow(torch.squeeze(torch.tensor(image)), 'gray')

    plt.pause(0.001)



fig = plt.figure()



for i in range(len(trainset)):

    sample = trainset[i]

    

#     print(i, sample['image'].shape, sample['label'].shape)



    ax = plt.subplot(1, 4, i + 1)

    ax.set_title('= {} ='.format(classes[sample['label']]))

    ax.axis('off')

    plt.tight_layout()

    show_imgs(**sample)

    



    if i == 3:

        plt.show()

        break
import torch.nn as nn

import torch.nn.functional as F



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 2)

        self.pool = nn.MaxPool2d(2)

        self.batch_norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 64, 3)

        self.batch_norm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3)

        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2048, 1024)

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(1024, 512)

        self.batch_norm4 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)

        self.fc4 = nn.Linear(256, 25)



    def forward(self, x):

        # correct shape

        x = x.reshape(-1, 1, 28, 28)

        # conv blocks

        x = self.pool(F.relu(self.conv1(x)))

        x = self.batch_norm1(x)

        x = self.swish(self.conv2(x))

        x = self.dropout(x)

        x = self.batch_norm2(x)

        x = self.pool(self.swish(self.conv3(x)))

        x = self.dropout(x)

        x = self.batch_norm3(x)

        # reshape for linear layers

        x = x.view(-1, 128*4*4)

        # linear block

        x = self.swish(self.fc1(x))

        x = self.dropout(x)

        x = self.swish(self.fc2(x))

        x = self.dropout(x)

        x = self.batch_norm4(x)

        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return x

        

    def swish(self, x):

        return x * F.sigmoid(x)



# net = Net()

# net(torch.randn(2, 1, 28, 28)).shape
from prettytable import PrettyTable



def count_parameters(model):

    table = PrettyTable(["Modules", "Parameters"])

    total_params = 0

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad: continue

        param = parameter.numel()

        table.add_row([name, param])

        total_params+=param

    print(table)

    print(f"Total Trainable Params: {total_params}")

    return total_params

    

count_parameters(Net())
if wb:

    wandb.init(project="sign-lang")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

steps = 0

print_every = 1

running_loss = 0

train_losses, test_losses = [], []





net = Net().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)



for epoch in range(epochs):

    for inputs in trainloader:

        steps += 1

        inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)

        optimizer.zero_grad()

        out = net(inputs)

        loss = criterion(out, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            net.eval()

            with torch.no_grad():

                for inputs in testloader:

                    inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)

                    out = net(inputs)

                    batch_loss = criterion(out, labels)

                    test_loss += batch_loss.item()

                    

                    ps = torch.exp(out)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            train_losses.append(running_loss/len(trainloader))

            test_losses.append(test_loss/len(testloader))                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(testloader):.3f}.. "

                  f"Test accuracy: {accuracy/len(testloader):.3f}")

            if wb:

                wandb.log({'Train Loss':running_loss/print_every, 'Test Loss': test_loss/len(testloader), 'Test Accuracy':accuracy/len(testloader)})



            running_loss = 0

            net.train()

dest = f'sign_lang_lr_{lr}_epo_{epochs}_'

v = 0

while os.path.isfile(dest+str(v)):

    v += 1

torch.save(net, dest+str(v)+'.pth')
# import torch.nn as nn

# import torch.nn.functional as F



# class XNet(nn.Module):

#     def __init__(self):

#         super(XNet, self).__init__()

#         self.conv1 = nn.Conv2d(1, 16, 2)

#         self.pool = nn.MaxPool2d(2)

#         self.batch_norm1 = nn.BatchNorm2d(16)

#         self.conv2 = nn.Conv2d(16, 64, 3)

#         self.batch_norm2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3)

#         self.batch_norm3 = nn.BatchNorm2d(128)

#         self.fc1 = nn.Linear(2048, 1024)

#         self.dropout = nn.Dropout(0.2)

#         self.fc2 = nn.Linear(1024, 512)

#         self.batch_norm4 = nn.BatchNorm1d(512)

#         self.fc3 = nn.Linear(512, 256)

#         self.fc4 = nn.Linear(256, 1)



#     def forward(self, x):

#         # correct shape

#         x = x.reshape(-1, 1, 28, 28)

#         # conv blocks

#         x = self.pool(F.relu(self.conv1(x)))

#         x = self.batch_norm1(x)

#         x = self.swish(self.conv2(x))

#         x = self.dropout(x)

#         x = self.batch_norm2(x)

#         x = self.pool(self.swish(self.conv3(x)))

#         x = self.dropout(x)

#         x = self.batch_norm3(x)

#         # reshape for linear layers

#         x = x.view(-1, 128*4*4)

#         # linear block

#         x = self.swish(self.fc1(x))

#         x = self.dropout(x)

#         x = self.swish(self.fc2(x))

#         x = self.dropout(x)

#         x = self.batch_norm4(x)

#         x = F.relu(self.fc3(x))

#         x = self.fc4(x)

#         return x

        

#     def swish(self, x):

#         return x * torch.sigmoid(x)



# # net = XNet()

# # net(torch.randn(2, 1, 28, 28))
# if wb:

#     wandb.init(project="sign-lang")



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# steps = 0

# print_every = 1

# running_loss = 0

# train_losses, test_losses = [], []





# net = XNet().to(device)

# criterion = nn.MSELoss()

# optimizer = torch.optim.Adam(net.parameters(), lr=lr)



# for epoch in range(epochs):

#     for inputs in trainloader:

#         steps += 1

#         inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)

#         optimizer.zero_grad()

#         out = net(inputs)

#         loss = criterion(torch.squeeze(out), labels.float())

#         loss.backward()

#         optimizer.step()

#         running_loss += loss.item()

        

#         if steps % print_every == 0:

#             test_loss = 0

#             accuracy = 0

#             net.eval()

#             with torch.no_grad():

#                 for inputs in testloader:

#                     inputs, labels = (inputs['image']).to(device), (inputs['label']).to(device)

#                     out = net(inputs)

#                     batch_loss = criterion(out, labels)

#                     test_loss += batch_loss.item()

                    

#                     ps = torch.exp(out)

#                     top_p, top_class = ps.topk(1, dim=1)

#                     equals = top_class == labels.view(*top_class.shape)

#                     accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

#             train_losses.append(running_loss/len(trainloader))

#             test_losses.append(test_loss/len(testloader))                    

#             print(f"Epoch {epoch+1}/{epochs}.. "

#                   f"Train loss: {running_loss/print_every:.3f}.. "

#                   f"Test loss: {test_loss/len(testloader):.3f}.. "

#                   f"Test accuracy: {accuracy/len(testloader):.3f}")

#             if wb:

#                 wandb.log({'Train Loss':running_loss/print_every, 'Test Loss': test_loss/len(testloader), 'Test Accuracy':accuracy/len(testloader)})



#             running_loss = 0

#             net.train()

# dest = f'Xsign_lang_lr_{lr}_epo_{epochs}'

# v = 0

# while os.path.isfile(dest+str(v)):

#     v += 1

# torch.save(net, dest+str(v)+'.pth')