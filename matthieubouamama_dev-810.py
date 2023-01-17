!pip install torchsummary
from IPython.core.interactiveshell import InteractiveShell
import seaborn as sns
import torchvision
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchsummary import summary
from timeit import default_timer as timer
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['font.size'] = 14
InteractiveShell.ast_node_interactivity = 'all'
import os
#le dataset Epitech et sa location
dataDir = '../input/epitech-810/dataset'
trainDir = dataDir + '/train/'
validDir = dataDir + '/val/'
testDir = dataDir + '/test/'
#pouvoir utiliser le gpu 
train_on_gpu = cuda.is_available()
multi_gpu = False
#les hyper paramètres
batchSize = 18
nbEpoch = 10

#data Augmentation
dataAugmentation = {
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data = {
    'train':
    datasets.ImageFolder(root=trainDir, transform=dataAugmentation['train']),
    'val':
    datasets.ImageFolder(root=validDir, transform=dataAugmentation['val']),
    'test':
    datasets.ImageFolder(root=testDir, transform=dataAugmentation['test'])
}

dataloaders = {
    'train': DataLoader(data['train'], batch_size=batchSize, shuffle=True),
    'val': DataLoader(data['val'], batch_size=batchSize, shuffle=True),
    'test': DataLoader(data['test'], batch_size=batchSize, shuffle=True)
}
trainiter = iter(dataloaders['train'])
features, labels = next(trainiter)
features.shape, labels.shape
# j'utilise le model vgg16 pour la detection d'image que je télécharge depuis le site pytorch
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
n_inputs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(256, 2), nn.LogSoftmax(dim=1))

model.classifier
#pour utiliser le gpu lors de l'apprentissage sinon on utilise .cpu()
model = model.to('cuda')

#liste des layers du model
summary(model, input_size=(3, 224, 224), batch_size=batchSize, device='cuda')
#créer les classes du model
model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}

list(model.idx_to_class.items())
#optimizer j'utilise Adam on peux aussi utilise SGD
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters())

for p in optimizer.param_groups[0]['params']:
    if p.requires_grad:
        print(p.shape)
def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          max_epochs_stop=3,
          n_epochs=1,
          print_every=1):
    """
    Params
    --------
        model (PyTorch model): le model
        criterion (PyTorch loss): criterion
        optimizer (PyTorch optimizier): optimizer
        train_loader (PyTorch dataloader): dataLoader train
        valid_loader (PyTorch dataloader): dataLoader valid
        max_epochs_stop (int): early stopping après max_epochs_stop
        n_epochs (int): nombre d'epoch variable a changer dans la section hyper parametre
        print_every (int): affichage des stat du model

    Returns
    --------
        model (PyTorch model): model entrainé
        history (DataFrame): history
    """

    # Early stopping
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    try:
        print(f'nombre d\'epochs: {model.epochs} epochs.\n')
    except:
        model.epochs = 0

    overall_start = timer()

    for epoch in range(n_epochs):

        # valid losses
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # train
        model.train()
        start = timer()
        for ii, (data, target) in enumerate(train_loader):
            # train sur le gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            output = model(data)

            # Loss
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()
            train_loss += loss.item() * data.size(0)

            # accurancy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)
            
            # print stats
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. temps écoulé: {timer() - start:.2f} secondes.',
                end='\r')

        # validation
        else:
            model.epochs += 1
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # validation loop
                for data, target in valid_loader:
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    valid_loss += loss.item() * data.size(0)

                    # accurancy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item() * data.size(0)

                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # print stat
                if (epoch + 1) % print_every == 0:
                    print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                    print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                if valid_loss < valid_loss_min:
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= max_epochs_stop:
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                        total_time = timer() - overall_start
                        print(f'temps écoulés: {total_time:.2f}. temps par epoch: {total_time / (epoch+1):.2f} secondes.')
                        model.optimizer = optimizer
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    total_time = timer() - overall_start
    print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
    print(f'temps écoulés: {total_time:.2f} secondes. temps par epoch: {total_time / (epoch):.2f} secondes.')
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history
#phase de train
model, history = train(
    model,
    criterion,
    optimizer,
    dataloaders['train'],
    dataloaders['val'],
    max_epochs_stop=5,
    n_epochs=nbEpoch,
    print_every=1)
#Resultat
plt.figure(figsize=(8, 6))
for c in ['train_loss', 'valid_loss']:
    plt.plot(history[c], label=c)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()
# Accuracy
plt.figure(figsize=(8, 6))
for c in ['train_acc', 'valid_acc']:
    plt.plot(
        100 * history[c], label=c)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()