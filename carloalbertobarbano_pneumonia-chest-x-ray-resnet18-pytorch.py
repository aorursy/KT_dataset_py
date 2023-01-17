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
!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL | wc -l

!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA | wc -l
!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL | wc -l

!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA | wc -l
!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL | wc -l

!ls -l /kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA | wc -l
!nvidia-smi
class ChestXRay(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):

        sample, target = super().__getitem__(index)

        path, _ = self.samples[index]

        

        target = 0

        if 'bacteria' in path:

            target = 1

        elif 'virus' in path:

            target = 2

        

        return sample, target

       
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



train_dataset = ChestXRay('/kaggle/input/chest-xray-pneumonia/chest_xray/train', transform=train_transforms)

val_dataset = ChestXRay('/kaggle/input/chest-xray-pneumonia/chest_xray/val', transform=transforms)

test_dataset = ChestXRay('/kaggle/input/chest-xray-pneumonia/chest_xray/test', transform=transforms)



train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
model = torchvision.models.resnet18(pretrained=True)

model.fc = torch.nn.Linear(in_features=512, out_features=3)

model = model.to(device)

model
def run_epoch(model, dataloader, criterion, optimizer, lr_scheduler, phase='train'):

    epoch_loss = 0.

    epoch_acc = 0.

    

    batch_num = 0.

    samples_num = 0.

    

    true_labels = []

    pred_labels = []

    

    for batch_idx, (data, labels) in enumerate(dataloader):

        data, labels = data.to(device), labels.to(device)

        

        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(data)

            _, preds = torch.max(outputs, 1)

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

optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.001)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
best_model = None

best_loss = 0.

best_test_loss = 0.

best_test_acc = 0.

best_pred_labels = []

true_labels = []



pred_labels = []

test_acc = 0.

test_loss = 0.
train_losses = []

val_losses = []

test_losses = []



for epoch in range(60):

    print('='*15, f'Epoch: {epoch}')

    

    train_loss, train_acc, _, _ = run_epoch(model, train_dataloader, criterion, optimizer, lr_scheduler)

    val_loss, val_acc, _, _ = run_epoch(model, val_dataloader, criterion, optimizer, lr_scheduler, phase='val')

    test_loss, test_acc, true_labels, pred_labels = run_epoch(model, test_dataloader, criterion, optimizer, lr_scheduler, phase='test')

    

    print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')

    print(f'Val loss: {val_loss}, Val accuracy: {val_acc}')

    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')

    print()

    

    train_losses.append(train_loss)

    val_losses.append(val_loss)

    test_losses.append(test_loss)

    

    torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet34-mooney-chest-x-ray-{seed}.pt')

    

    if best_model is None or val_loss < best_loss:

        best_model = copy.deepcopy(model)

        best_loss = val_loss

        best_test_loss = test_loss

        best_test_acc = test_acc 

        best_pred_labels = pred_labels

        torch.save({'epoch': epoch, 'model': model.state_dict()}, f'resnet34-mooney-chest-x-ray-best-{seed}.pt')

        
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



plt.figure(figsize=(18, 8))

plt.plot(train_losses, label='Train loss')

plt.plot(val_losses, label='Val loss')

plt.plot(test_losses, label='Test loss')

plt.legend()

plt.show()
print(f'Best model test accuracy: {best_test_acc}')

print(f'Best model test loss: {best_test_loss}')



print(f'Final model test accuracy: {test_acc}')

print(f'Final model test loss: {test_loss}')
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(true_labels, best_pred_labels)

#tn, fp, fn, tp = cm.ravel()

ax = sns.heatmap(cm, annot=True, fmt="d")
#recall = tp/(tp + fn)

#precision = tp/((tp + fp))

#f1_score = 2 * (recall * precision)/(precision + recall)

#print("Best model F1 Score:", f1_score)
cm = confusion_matrix(true_labels, pred_labels)

#tn, fp, fn, tp = cm.ravel()

ax = sns.heatmap(cm, annot=True, fmt="d")
#recall = tp/(tp + fn)

#precision = tp/((tp + fp))

#f1_score = 2 * (recall * precision)/(precision + recall)

#print("Final model F1 Score:", f1_score)