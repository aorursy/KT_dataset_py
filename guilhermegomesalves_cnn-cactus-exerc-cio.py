import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

plt.style.use("ggplot")



# OpenCV Image Library

import cv2



# Import PyTorch

import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset

import torchvision

import torch.optim as optim
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")

train_df.head()
from zipfile import ZipFile

train = "/kaggle/input/aerial-cactus-identification/train.zip"

with ZipFile(train, "r") as extra:

    #extra.printdir()

    extra.extractall()
from zipfile import ZipFile

test = "/kaggle/input/aerial-cactus-identification/test.zip"

with ZipFile(test, "r") as extra:

    #extra.printdir()

    extra.extractall()

print(f"Train Size: {len(os.listdir('train'))}")

print(f"Test Size: {len(os.listdir('test'))}")

value_counts = train_df.has_cactus.value_counts()

plt.pie(value_counts, labels=['Has Cactus', 'No Cactus'], autopct='%1.1f', colors=['green', 'red'], shadow=True)

plt.figure(figsize=(5,5))

plt.show()
train_path = 'train/'

test_path = 'test/'
# Definindo uma classe para o conjuntos de dados



class CreateDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
transforms_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



train_data = CreateDataset(df_data=train_df, data_dir=train_path, transform=transforms_train)
# Definir tamanho do lote

batch_size = 64



# Porcentagem de treinamento definida para uso como validação

valid_size = 0.2



# obter índices de treinamento que serão usados para validação

num_train = len(train_data)

indices = list(range(num_train))

np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]



# Criar amostras

train_sampler = SubsetRandomSampler(train_idx)

valid_sampler = SubsetRandomSampler(valid_idx)



# preparar carregadores de dados (combinar conjunto de dados e amostrador)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)

valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
transforms_test = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



# criando dados de teste

sample_sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/sample_submission.csv")

test_data = CreateDataset(df_data=sample_sub, data_dir=test_path, transform=transforms_test)



# carregando dados de teste

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
classes = [ 'No Cactus','Cactus']
def imshow(img):

    '''Helper function to un-normalize and display an image'''

    # unnormalize

    img = img / 2 + 0.5

    # convert from Tensor image and display

    plt.imshow(np.transpose(img, (1, 2, 0)))
# obter um lote de imagens de treinamento

dataiter = iter(train_loader)

images, labels = dataiter.next()

images = images.numpy() # converter imagens em numpy para exibição



# plota as imagens no lote, juntamente com os rótulos correspondentes

fig = plt.figure(figsize=(25, 4))

# exibir 20 imagens

for idx in np.arange(20):

    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])

    ax.set_title(classes[labels[idx]])
rgb_img = np.squeeze(images[3])

channels = ['red channel', 'green channel', 'blue channel']



fig = plt.figure(figsize = (36, 36)) 

for idx in np.arange(rgb_img.shape[0]):

    ax = fig.add_subplot(3, 1, idx + 1)

    img = rgb_img[idx]

    ax.imshow(img, cmap='gray')

    ax.set_title(channels[idx])

    width, height = img.shape

    thresh = img.max()/2.5

    for x in range(width):

        for y in range(height):

            val = round(img[x][y],2) if img[x][y] !=0 else 0

            ax.annotate(str(val), xy=(y,x),

                    horizontalalignment='center',

                    verticalalignment='center', size=8,

                    color='white' if img[x][y]<thresh else 'black')
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        # Convolutional Layer (sees 32x32x3 image tensor) 

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Convolutional Layer (sees 16x16x16 image tensor)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Convolutional Layer (sees 8x8x32 image tensor)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Convolutional Layer (sees 4*4*64 image tensor)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        # Maxpooling Layer

        self.pool = nn.MaxPool2d(2, 2)

        # Linear Fully-Connected Layer 1 (sees 2*2*128 image tensor)

        self.fc1 = nn.Linear(128*2*2, 512)

        # Linear FC Layer 2

        self.fc2 = nn.Linear(512, 2)

        # Set Dropout

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, x):

        # add sequence of convolutional and max pooling layers

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = self.pool(F.relu(self.conv4(x)))

        # flatten image input

        x = x.view(-1, 128 * 2 * 2)

        # add dropout layer

        x = self.dropout(x)

        # add 1st hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add 2nd hidden layer, with relu activation function

        x = self.fc2(x)

        return x

# check if CUDA is available

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
model = CNN()

print(model)



# Move model to GPU if available

if train_on_gpu: model.cuda()
# specify loss function (categorical cross-entropy loss)

criterion = nn.CrossEntropyLoss()



# specify optimizer

optimizer = optim.Adamax(model.parameters(), lr=0.001)
# number of epochs to train the model

n_epochs = 30



valid_loss_min = np.Inf # track change in validation loss



# keeping track of losses as it happen

train_losses = []

valid_losses = []



for epoch in range(1, n_epochs+1):



    # keep track of training and validation loss

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # train the model #

    ###################

    model.train()

    for data, target in train_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # clear the gradients of all optimized variables

        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters

        loss.backward()

        # perform a single optimization step (parameter update)

        optimizer.step()

        # update training loss

        train_loss += loss.item()*data.size(0)

        

    ######################    

    # validate the model #

    ######################

    model.eval()

    for data, target in valid_loader:

        # move tensors to GPU if CUDA is available

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # forward pass: compute predicted outputs by passing inputs to the model

        output = model(data)

        # calculate the batch loss

        loss = criterion(output, target)

        # update average validation loss 

        valid_loss += loss.item()*data.size(0)

    

    # calculate average losses

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

        

    # print training/validation statistics 

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # save model if validation loss has decreased

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'best_model.pt')

        valid_loss_min = valid_loss
%config InlineBackend.figure_format = 'retina'



plt.plot(train_losses, label='Training loss')

plt.plot(valid_losses, label='Validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(frameon=False)
# Carregar Os melhores parâmetros aprendidos com o treinamento em nosso modelo para fazer previsões mais tarde

model.load_state_dict (torch.load ('best_model.pt'))
# Turn off gradients

model.eval()



preds = []

for batch_i, (data, target) in enumerate(test_loader):

    #data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)



# Create Submission file        

sample_sub['has_cactus'] = preds

sample_sub.to_csv('submission.csv', index=False)

#sample_sub = pd.read_csv("/kaggle/input/aerial-cactus-identification/submission.csv", index=False)
os.listdir()