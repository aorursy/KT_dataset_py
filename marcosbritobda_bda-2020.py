import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir('../input/'))



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
train_df = pd.read_csv("../input/aerial-cactus-identification/train.csv")

train_df.head()
print(f"Train Size: {len(os.listdir('../input/aerial-cactus-identification/train/train'))}")

print(f"Test Size: {len(os.listdir('../input/aerial-cactus-identification/test/test'))}")
# Contando o número de dados de amostra para cada classe

value_counts = train_df.has_cactus.value_counts()

%matplotlib inline

plt.pie(value_counts, labels=['Has Cactus', 'No Cactus'], autopct='%1.1f', colors=['green', 'red'], shadow=True)

plt.figure(figsize=(5,5))

plt.show()
# Caminhos de dados

train_path = '../input/aerial-cactus-identification/train/train/'

test_path = '../input/aerial-cactus-identification/test/test/'
# Nossa própria classe personalizada para conjuntos de dados

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



# Criar Samplers

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

sample_sub = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

test_data = CreateDataset(df_data=sample_sub, data_dir=test_path, transform=transforms_test)



# prepare o carregador de teste

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
classes = [ 'No Cactus','Cactus']
def imshow(img):

    '''Helper function to un-normalize and display an image'''

    # anormalizar

    img = img / 2 + 0.5

    # converter da imagem Tensor e exibir

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

        # Camada Convolucional (veja o tensor de imagem 32x32x3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        # Camada convolucional (veja o tensor da imagem 16x16x16)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Camada convolucional (consulte o tensor de imagem 8x8x32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        # Camada Convolucional (veja o tensor de imagem 4 * 4 * 64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)

        # Camada Maxpooling

        self.pool = nn.MaxPool2d(2, 2)

        # Camada 1 linear totalmente conectada (consulte o tensor de imagem 2 * 2 * 128)

        self.fc1 = nn.Linear(128*2*2, 512)

        # Camada FC linear 2

        self.fc2 = nn.Linear(512, 2)

        # Definir desistência

        self.dropout = nn.Dropout(0.2)

        

    def forward(self, x):

        # adicionar sequência de camadas convolucionais e de max pooling

        x = self.pool(F.relu(self.conv1(x)))

        x = self.pool(F.relu(self.conv2(x)))

        x = self.pool(F.relu(self.conv3(x)))

        x = self.pool(F.relu(self.conv4(x)))

        # achatar a entrada da imagem

        x = x.view(-1, 128 * 2 * 2)

        # adicionar camada de desistência

        x = self.dropout(x)

        # adicione a 1ª camada oculta, com função de ativação relu

        x = F.relu(self.fc1(x))

        # adicionar camada de desistência

        x = self.dropout(x)

        # adicione a segunda camada oculta, com função de ativação relu

        x = self.fc2(x)

        return x
# verifique se CUDA está disponível

train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('CUDA is not available.  Training on CPU ...')

else:

    print('CUDA is available!  Training on GPU ...')
# crie uma CNN completa

model = CNN()

print(model)



# Mover modelo para GPU, se disponível

if train_on_gpu: model.cuda()
# especificar função de perda (perda de entropia cruzada categórica)

criterion = nn.CrossEntropyLoss()



# especificar otimizador

optimizer = optim.Adamax(model.parameters(), lr=0.001)
# número de épocas para treinar o modelo

n_epochs = 40



valid_loss_min = np.Inf # rastrear alteração na perda de validação



# acompanhando as perdas assim que acontecem

train_losses = []

valid_losses = []



for epoch in range(1, n_epochs+1):



    # acompanhar as perdas de treinamento e validação

    train_loss = 0.0

    valid_loss = 0.0

    

    ###################

    # treinar o modelo#

    ###################

    model.train()

    for data, target in train_loader:

        # mover tensores para GPU se CUDA estiver disponível

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

        # limpe os gradientes de todas as variáveis otimizadas

        optimizer.zero_grad()

        # forward forward: calcula as saídas previstas passando entradas para o modelo

        output = model(data)

        # calcular a perda de lote

        loss = criterion(output, target)

        #retrocesso: calcular o gradiente da perda em relação aos parâmetros do modelo

        loss.backward()

        # execute uma única etapa de otimização (atualização de parâmetro)

        optimizer.step()

        # atualizar perda de treinamento

        train_loss += loss.item()*data.size(0)

        

    ######################    

   # validar o modelo #

    ######################

    model.eval()

    for data, target in valid_loader:

        # mova tensores para a GPU se CUDA estiver disponível

        if train_on_gpu:

            data, target = data.cuda(), target.cuda()

       # forward pass: calcula as saídas previstas passando entradas para o modelo

        output = model(data)

        # calcular a perda de lote

        loss = criterion(output, target)

       # atualizar perda média de validação

        valid_loss += loss.item()*data.size(0)

    

   # calcular perdas médias

    train_loss = train_loss/len(train_loader.sampler)

    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

        

   # imprimir estatísticas de treinamento / validação

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    

    # salvar modelo se a perda de validação diminuiu

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

        valid_loss_min,

        valid_loss))

        torch.save(model.state_dict(), 'best_model.pt')

        valid_loss_min = valid_loss
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



plt.plot(train_losses, label='Training loss')

plt.plot(valid_losses, label='Validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(frameon=False)
# Carregar Os melhores parâmetros aprendidos com o treinamento em nosso modelo para fazer previsões mais tarde

model.load_state_dict(torch.load('best_model.pt'))
# Desativar gradientes

model.eval()



preds = []

for batch_i, (data, target) in enumerate(test_loader):

    data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)



# Criar arquivo de envio     

sample_sub['has_cactus'] = preds

sample_sub.to_csv('submission.csv', index=False)