import torch

import torch.nn as nn

import torchvision.datasets as datasets

import matplotlib.pyplot as plt

import numpy as np

import os

from PIL import Image

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from torch.nn.init import kaiming_normal_ as he_init

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
!rm -rf data

!mkdir data

!wget -P data -O data/concrete_data.zip https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip

!unzip -q -d data data/concrete_data.zip
# class Data(Dataset):

#     def __init__(self, transforms, train = False):

#         self.x = torch.zeros(1)

#         self.y = torch.zeros(1)

        

#         negative_file_path = os.path.join('concrete_class/', 'Negative')

#         negative_files = [file for file in os.listdir(negative_file_path) if file.endswith('.jpg')]

#         negative_files.sort()

#         positive_file_path = os.path.join('concrete_class/', 'Positive')

#         positive_files = [file for file in os.listdir(positive_file_path) if file.endswith('.jpg')]

#         positive_files.sort()

#         print(negative_files[0], type(negative_files[0]))

        

#         if train:

#             self.x = torch.zeros(36000, 1)

#             for i in range(36000):

#                 self.x[i] = transforms(negative_files[i]) if i < 18000 else transforms(negative_files[i])

#             self.y = torch.zeros(36000, 1, dtype = 'uint8')

#             self.y[18000:, :] = 1

#         else:

#             self.x = torch.zeros(4000, 1)

#             for i in range(4000):

#                 self.x[i] = transforms(negative_files[i + 36000]) if i < 2000 else transforms(negative_files[i + 38000])

#             self.y = torch.zeros(4000, 1, dtype = 'uint8')

#             self.y[2000:, :] = 1

        

#     def __getitem__(self, index):

#         return self.y[index], self.x[index]

#     def __len__(self):

#         return self.y.shape[0]
class CNN(nn.Module):

    def __init__(self, input_channels = 3, out_1 = 16, out_2 = 32, out_3 = 64):

        super(CNN, self).__init__()

        

        self.conv_l1 = nn.Sequential(

            nn.Conv2d(in_channels = input_channels, out_channels = out_1, kernel_size = 10), # Changed input_channels

#             nn.Dropout2d(p = 0.35, inplace = True),

            nn.MaxPool2d(kernel_size = 8, stride = 2),

            nn.BatchNorm2d(out_1, momentum = 0.8),

            nn.ReLU()

        )

        

        self.conv_l2 = nn.Sequential(

            nn.Conv2d(in_channels = out_1, out_channels = out_2, kernel_size = 8),

#             nn.Dropout2d(p = 0.35, inplace = True),

            nn.MaxPool2d(kernel_size = 6, stride = 2),

            nn.BatchNorm2d(out_2, momentum = 0.8),

            nn.ReLU()

        )

        

        self.conv_l3 = nn.Sequential(

            nn.Conv2d(in_channels = out_2, out_channels = out_3, kernel_size = 6),

#             nn.Dropout2d(p = 0.35, inplace = True),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.BatchNorm2d(out_3, momentum = 0.8),

            nn.ReLU()

        )

        # Added conv_l4

        self.conv_l4 = nn.Sequential(

            nn.Conv2d(in_channels = out_3, out_channels = 96, kernel_size = 4),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

#             nn.Dropout2d(p = 0.35, inplace = True),

            nn.BatchNorm2d(96, momentum = 0.4),

            nn.ReLU(),

            nn.Flatten()

        )

        

        self.fully_connected = nn.Sequential(

            nn.Linear(384, 150),

            nn.Dropout(p = 0.35, inplace = True),

            nn.ReLU(),

            nn.Linear(150, 100),

            nn.Dropout(p = 0.35, inplace = True),

            nn.Linear(100, 100),

            nn.ReLU(),

#             nn.Dropout(p = 0.1, inplace = True),

#             nn.ReLU(),

            nn.Linear(100, 10)

        )



    def forward(self, x):

        x = self.conv_l1(x)

        x = self.conv_l2(x)

        x = self.conv_l3(x)

        x = self.conv_l4(x)

        x = self.fully_connected(x)

        return x
if torch.cuda.is_available():

    device = torch.device('cuda:0')

    print(torch.cuda.get_device_name(0), torch.cuda.current_device())

else:

    device = torch.device('cpu')

device
def train(model, criterion, optimizer, train_loader, val_loader, epochs):

    loss_acc = {'training_loss': [], 'training_acc': [], 'validation_loss': [], 'validation_acc': []}

    for epoch in range(epochs):

        tr_acc = 0

        tr_loss = 0

        

        print('Training epoch: {}...'.format(epoch))

        for i, (x, y) in tqdm(enumerate(train_loader)):

            optimizer.zero_grad()

            z = model.forward(x)

            loss = criterion(z, y)

            loss.backward()

            optimizer.step()

            _, y_hat = torch.max(z, 1)

            tr_acc += (y_hat == y).sum().item()

            tr_loss += loss.data

            loss_acc['training_loss'].append(loss.data)

            loss_acc['training_acc'].append(100 * (tr_acc / (i + 1)))

            

        val_acc = 0

        print('Validating...')

        for i, (x, y) in tqdm(enumerate(val_loader)):

            z = model.forward(x)

            loss = criterion(z, y)

            _, y_hat = torch.max(z, 1)

            val_acc += (y_hat == y).sum().item()

            loss_acc['validation_loss'].append(loss.data)

            loss_acc['validation_acc'].append(100 * (val_acc / (i + 1)))

        

        print('Epoch: {} mean results, training loss: {}, validation accuracy: {}\n'

              .format(epoch + 1, tr_loss / len(train_loader), 100 * (val_acc / len(val_loader))))

    return loss_acc
def init_weights(layer):

    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:

        he_init(layer.weight)
class ToCudaTensor(transforms.ToTensor):

    def __call__(self, image):

        img = transforms.ToTensor()

        img = img(image)

        img = img.float().to(0)

        return img

    def __repr__(self):

        return self.__class__.__name__ + '()'





TransformTargets = lambda x: torch.tensor(x).to(device)
from torch import multiprocessing



try:

    multiprocessing.set_start_method('spawn')

except RuntimeError:

    print('Unable to execute spawn:', RuntimeError)



transformer = transforms.Compose([transforms.Resize((120, 120)), ToCudaTensor(), torch.nn.Sigmoid()]) # Got rid of grayscale, temporarily

images = datasets.ImageFolder(root = 'data', transform = transformer, target_transform = TransformTargets) #, transform = ToCudaTensor())

# images = datasets.ImageFolder(root = 'data', transform = transforms.ToTensor())



train_data, val_data = torch.utils.data.random_split(images, [36000, 4000])



train_loader = DataLoader(train_data)#, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = True)

val_loader = DataLoader(val_data)#, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = True)
np.shape(images[0][0]), images[0]
images.classes, type(train_data), type(val_data)
model = CNN()

model = model.cuda(0)

model = model.to(device)

model.apply(init_weights)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters())

epochs = 30

model.modules
model_data = train(model, criterion, optimizer, train_loader, val_loader, epochs)
plt.plot(model_data['training_loss'], label = 'loss')

plt.plot(model_data['training_acc'], label = 'accuracy')

plt.xticks([i + 1 for i in range(epochs)])

plt.ylim(0, 100)

plt.title('Training')

plt.xlabel('Epochs')

plt.legend()

plt.show()
plt.plot(model_data['validation_loss'], label = 'loss')

plt.plot(model_data['validation_acc'], label = 'accuracy')

plt.xticks([i + 1 for i in range(epochs)])

plt.ylim(0, 100)

plt.title('Validation')

plt.xlabel('Epochs')

plt.legend()

plt.show()
torch.save(model.state_dict(), 'model')