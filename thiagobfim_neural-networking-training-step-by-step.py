# Imports here
import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

%matplotlib inline
import os
print(os.listdir("../input/flower-data/flower_data/flower_data"))
#Para carregar é necessário que já o dataset encontre-se no mesmo diretorio que esse arquivo.
data_dir = '../input/flower-data/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#É necessário carregar os dados de teste e de treino, e realizar uma transformação dos dados para deixar todos no mesmo padrão.
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

# Batch_size é o numero de elementos que será carregado a cada iteração,
#isso serve para não carregar todos os dados de uma unica vez e deixar o treinamento do modelo muito lerdo.
batch_size = 20
num_workers=0

# Prepara os DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

import json

with open('../input/labels/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

classes = list(cat_to_name.values())

# Obtem um batch de imagens, que foi definido como 20 no primeiro passo.
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # Converte imagens para exibir com numpy

# Imprimi as imagens no batch de imagens com a label correspondente.
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)).astype(np.uint8))
    ax.set_title(classes[labels[idx]])
#Nesse caso foi utilizado a rede resnet18 por não ser muito pesada e por conseguir fornecer um bom resultado nesse caso.
redeNeural = models.resnet18(pretrained=True)
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
##Alterar a quantidade de ouput conforme o dataset escolhido, nesse caso 102
from collections import OrderedDict
import torch.nn as nn
n_inputs = redeNeural.fc.in_features   
last_layer = nn.Linear(n_inputs, len(classes))
redeNeural.fc = last_layer
#Antes de validar o modelo é preciso incluir o tipo de criterion e optimizer

import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(redeNeural.fc.parameters(), lr=0.001)
if train_on_gpu:
    redeNeural.cuda()
test_loss = 0.0
size = len(classes);
class_correct = list(0. for i in range(size))
class_total = list(0. for i in range(size))

redeNeural.eval() # prep model for evaluation
test_losses = []

for batch_i, (data, target) in enumerate(test_loader):
    # forward pass: compute predicted outputs by passing inputs to the model
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = redeNeural(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(pred.size()[0]):
        try:
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
        except:
            print("Não foi encontrado o dataset de teste para a classe: ", classes[i])
            print("Não foi encontrado o dataset de teste para a classe: ", torch.max(output))
            print("Não foi encontrado o dataset de teste para a classe: ", torch.max(output,1))

redeNeural.train()
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
test_losses.append(test_loss/len(test_loader))
print('Test Loss: {:.6f}\n'.format(test_loss))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# Epoch é algo semelhante a iteração no caso de ML(Machine Learning). 
# Quanto mais Epoch mais o modelo será treinado e terá um desempenho melhor. Mas é necessário tomar cuidado com Overfitting.
n_epochs = 10
globalLoss = 100;
for epoch in range(1, n_epochs+1):

    # A perda de treino será incrementada a cada batch 
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    for batch_i, (data, target) in enumerate(train_loader):

        ##Utilizar GPU caso esteja disponivel.
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
            
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = redeNeural(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        if batch_i % 20 == 19:    # print training loss every specified number of mini-batches
            train_loss_mean = train_loss / 20;
            print('Epoch %d, Batch %d loss: %.16f' % (epoch, batch_i + 1, train_loss_mean))
            if  train_loss_mean <= globalLoss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                globalLoss,
                train_loss_mean))
                torch.save(redeNeural.state_dict(), 'model-saved.pth')
                globalLoss = train_loss_mean
            train_loss = 0.0
## Método para reccaregar o modelo já treinado
def _reload_module():
    redeNeural.load_state_dict(torch.load('model-saved.pth'))
# Reccarega o modelo treinado.
_reload_module()
test_loss = 0.0
size = len(classes);

class_correct = list(0. for i in range(size))
class_total = list(0. for i in range(size))

redeNeural.eval() # prep model for evaluation
test_losses = []

for batch_i, (data, target) in enumerate(test_loader):
    # forward pass: compute predicted outputs by passing inputs to the model
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = redeNeural(data)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        try:
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
        except:
            print(target.size())
            print("Não foi encontrado o dataset de teste para a classe: ", classes[i])

redeNeural.train()
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
test_losses.append(test_loss/len(test_loader))
print('Test Loss: {:.6f}\n'.format(test_loss))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()


# move model inputs to cuda, if GPU available
if train_on_gpu:
    imagesCuda = images.cuda()
    redeNeural.cuda()
output = redeNeural(imagesCuda)

# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

images = images.numpy()

# plot the images in the batch, along with predicted and true labels
count = 0
fig = plt.figure(figsize=(37, 8))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)).astype(np.uint8))
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
    if preds[idx].item()==labels[idx].item():
        count +=1
print("Acertos: ", count)
