#Universidade Salvador

#Disciplina:Automação Industrial

#Docente:Caio Viturino

#Discentes:Danielle Brito, Icaro Trindade e Robson Barbosa

#Importando o dataset

import torch

from torch.utils.data import Dataset, DataLoader

from torch import optim, nn

import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets, transforms

import numpy as np

import helper

import torch.nn.functional as F
%matplotlib inline

%config InlineBackend.figure_format='retina'
#Definir transformações básicas de nomalização para as imagens.

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#Download dos datasets de treino e de teste.



##Dados de treinamento

train_data=torchvision.datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=True,transform=transform)

trainset=train_data

##Dados de teste

test_data=datasets.FashionMNIST('~/.pytorch/F_MNIST_data/',download=True,train=False,transform=transform)

testset=test_data
#Carregar os dados dentro das variáveis

#Determinar o número de imagens por interação

#Randomização do dataset nas interações de treino



##Dados de treinamento

trainloader=torch.utils.data.DataLoader(dataset=trainset, batch_size=64, shuffle=True)



##Dados de teste

testloader=torch.utils.data.DataLoader(dataset=testset, batch_size=64, shuffle=True)
#Construção da rede

#As imagens possuem dimensão 28x28, tendo assim 784 pixels

#Serão usados 784 neurônios de entrada

#Serão usadas duas camadas ocultas,implementação abaixo



input_size = 784

hidden_size = [128, 64]

output_size = 10



#Usando a função de ativação ReLu e a fução log-softmax para retorno da rede

model = nn.Sequential(nn.Linear(input_size, hidden_size[0]),

                      nn.ReLU(),

                      nn.Linear(hidden_size[0], hidden_size[1]),

                      nn.ReLU(),

                      nn.Linear(hidden_size[1], output_size),

                      nn.LogSoftmax(dim=1))
#Treinamento da rede



#Definindo a função de perda e a otimização da rede

#Função de otimização SGD

#Função de perda:negative log likelihood loss

#O parametro lr é a taxa de aprendizagem

criterion=nn.NLLLoss()

optimizer=torch.optim.SGD(model.parameters(), lr=0.03)



#Após a definição dos parametros, treinamento efetivo da rede

#Calculo de erro e reajuste dos pesos

epochs=15

train_losses,test_losses=[],[]



for e in range(epochs):

    running_loss=0

    for images, labels in trainloader:

        images=images.view(images.shape[0],-1)



        optimizer.zero_grad()

        logps=model(images)

        loss=criterion(logps,labels)

        

        loss.backward()

        optimizer.step()

        

        running_loss+=loss.item()

    else:

        test_loss=0

        accuracy=0
#Validação da rede



with torch.no_grad():

    for test_images, test_labels in testloader:

        test_images=test_images.view(test_images.shape[0],-1)

        

        logps=model(test_images)

        test_loss+=criterion(logps, test_labels)

        

        ps=torch.exp(logps)

        

        top_p,top_class=ps.topk(1, dim=1)

        

        equals=top_class==test_labels.view(*top_class.shape)

        accuracy+= torch.mean(equals.type(torch.FloatTensor))

        

        train_losses.append(running_loss/len(trainloader))

        test_losses.append(test_loss/len(testloader))

        

        

        print("Epoch {}/{}..".format(e+1,epochs),

              "Training loss: {:.3f}..".format(train_losses[-1]),

              "Test loss: {:.3f}..".format(test_losses[-1]),

              "Test Accuracy: {:.3f}%".format(accuracy/len(testloader)))
#Gráfico de teste e validação da rede

plt.plot(train_losses,label='Training loss')

plt.plot(test_losses,label='Validation loss')

plt.legend(frameon=False)
#Funções para a impressão das imagens

def imshow(image, ax=None, title=None, normalize=True):

    """Imshow for Tensor."""

    if ax is None:

        fig, ax = plt.subplots()

    image = image.numpy().transpose((1, 2, 0))



    if normalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        image = np.clip(image, 0, 1)



    ax.imshow(image)

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_visible(False)

    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='both', length=0)

    ax.set_xticklabels('')

    ax.set_yticklabels('')



    return ax



def view_classify(img, ps, version="MNIST"):

    ''' Function for viewing an image and it's predicted classes.

    '''

    ps = ps.data.numpy().squeeze()



    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)

    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())

    ax1.axis('off')

    ax2.barh(np.arange(10), ps)

    ax2.set_aspect(0.1)

    ax2.set_yticks(np.arange(10))

    if version == "MNIST":

        ax2.set_yticklabels(np.arange(10))

    elif version == "Fashion":

        ax2.set_yticklabels(['T-shirt/top',

                            'Trouser',

                            'Pullover',

                            'Dress',

                            'Coat',

                            'Sandal',

                            'Shirt',

                            'Sneaker',

                            'Bag',

                            'Ankle Boot'], size='small');

    ax2.set_title('Class Probability')

    ax2.set_xlim(0, 1.1)



    plt.tight_layout()

#Testando a rede



dataiter=iter(testloader)

images,labesl= dataiter.next()

img=images[0]



#Convertendo a imagem em 2D em um vetor 1D

img=img.resize_(1,784)



#Calculando as probabilidades

with torch.no_grad():

    logps=model(img)



ps=torch.exp(logps)



#Imprimindo a imagens e as probabilidades

view_classify(img.resize_(1,28,28),ps,version='Fashion')