import torch 

import torchvision

import numpy as np 

import random

import os

import glob

import copy



def set_seed(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)



seed = 42

set_seed(seed)

device = torch.device('cuda:0')
!ls -l /kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test\(color\)/Test/
train_transforms = torchvision.transforms.Compose([

    torchvision.transforms.Resize(256),

    torchvision.transforms.RandomAffine(0, translate=(0, 0.1), scale=(1, 1.10)),

    torchvision.transforms.CenterCrop(224),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])



transforms = torchvision.transforms.Compose([

    torchvision.transforms.Resize(256),

    torchvision.transforms.CenterCrop(224),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

])



train_dataset = torchvision.datasets.ImageFolder("/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/color/", transform=train_transforms)

test_dataset = torchvision.datasets.ImageFolder("/kaggle/input/crowdai-plant-disease-dataset/Custom-Train-Test(color)/Test/", transform=transforms)



train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
len(train_dataset.classes)
train_dataset.classes
train_dataset.class_to_idx
class PlantDiseaseClassifier(torch.nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        

        encoder = torchvision.models.resnet18(pretrained=pretrained)

        encoder = torch.nn.Sequential(*list(encoder.children())[:-1])

        for param in encoder.parameters():

            param.requires_grad = False

        

        self.encoder = encoder

        self.classifier = torch.nn.Linear(in_features=512, out_features=25)

    

    def forward(self, x):

        x = self.encoder(x)

        x = torch.flatten(x, 1)

        return self.classifier(x)
model = PlantDiseaseClassifier(pretrained=True).to(device)

model
def run_epoch(model, dataloader, criterion, optimizer, lr_scheduler, phase='train'):

    epoch_loss = 0.

    epoch_acc = 0.

    

    batch_num = 0.

    samples_num = 0.

    

    true_labels = []

    pred_labels = []

    

    if phase == 'train':

        model.train()

    else:

        model.eval()

    

    for batch_idx, (data, labels) in enumerate(dataloader):

        data, labels = data.to(device), labels.to(device)

        

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(data)

            _, preds = torch.max(torch.nn.Softmax()(outputs), 1)

            loss = criterion(outputs, labels)

        

        true_labels.append(labels.detach().cpu())

        pred_labels.append(preds.detach().cpu())

        

        if phase == 'train':

            loss.backward()

            optimizer.step()

        

        print(f'\r{phase} batch [{batch_idx}/{len(dataloader)}]: loss {loss.item()}', end='', flush=True)

        epoch_loss += loss.detach().cpu().item()

        epoch_acc += torch.sum(preds == labels.data)

        batch_num += 1

        samples_num += len(labels)

    

    print()

    return epoch_loss / batch_num, epoch_acc / samples_num, torch.cat(true_labels).numpy(), torch.cat(pred_labels).numpy()
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
true_labels = []

pred_labels = []

test_acc = 0.

test_loss = 0.
train_losses = []

test_losses = []



for epoch in range(20):

    print('='*15, f'Epoch: {epoch}')

    

    train_loss, train_acc, _, _ = run_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler)

    test_loss, test_acc, true_labels, pred_labels = run_epoch(model, test_dataloader, criterion, optimizer, lr_scheduler, phase='test')

    

    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')

    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

    print()

    

    train_losses.append(train_loss)

    test_losses.append(test_loss)

    

    torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet18-plantvillage-{seed}.pt')

    

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



plt.figure(figsize=(18, 8))

plt.plot(train_losses, label='Train loss')

plt.plot(test_losses, label='Test loss')

plt.legend()

plt.show()
print(f'Final model test accuracy: {test_acc}')

print(f'Final model test loss: {test_loss}')
from sklearn.metrics import confusion_matrix



plt.figure(figsize=(15, 14))

cm = confusion_matrix(true_labels, pred_labels)

ax = sns.heatmap(cm, annot=True, fmt="d")