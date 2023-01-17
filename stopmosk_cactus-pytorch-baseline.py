import random

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import os

from os.path import join

import shutil



from tqdm import tqdm   # Progress bar



import torch

import torchvision

import torch.nn.functional as T

from torchvision import transforms, models

from torch.utils.data import DataLoader, Dataset





random.seed(6)

np.random.seed(6)

torch.manual_seed(6)

torch.cuda.manual_seed(6)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False



root_dir = ''

input_dir = '../input/aerial-cactus-identification'
train_val_labels = pd.read_csv(join(input_dir, 'train.csv'))

train_val_labels.head()
plt.figure(figsize=(3,3))

plt.title('Labels distribution')

sns.countplot(train_val_labels['has_cactus']);
labels = ['no_cactus', 'has_cactus']



train_dir = join(root_dir, 'train')

val_dir = join(root_dir, 'val')

test_dir = join(root_dir, 'test')



# Make train and val folders

for label in labels:

    os.makedirs(join(train_dir, label), exist_ok=True)

    os.makedirs(join(val_dir, label), exist_ok=True)
# 15000 train photos and 2500 val photos



source_dir = join(input_dir, 'train', 'train')



for i, filename in enumerate(tqdm(os.listdir(source_dir))):



#     if i % 10 != 0:   # Skip 90% photos

#         continue



    is_cactus = int(train_val_labels.loc[train_val_labels['id'] == filename]['has_cactus'])



    if i % 7 == 0:   #if i % 7 == 0:   

        shutil.copy(join(source_dir, filename), join(val_dir, labels[is_cactus], filename))

    else:

        shutil.copy(join(source_dir, filename), join(train_dir, labels[is_cactus], filename))
def show_sample_images(dataloader, batch_size, images_from_batch=0, denormalize=False, classes=None):

    if denormalize:

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

    else:

        mean = np.array([0., 0., 0.])

        std = np.array([1., 1., 1.])

    

    if images_from_batch == 0 or images_from_batch > batch_size:

            images_from_batch = batch_size

        

    for images, labels in dataloader:

        plt.figure(figsize=(20, (batch_size // 20 + 1) * 3))



        cols = 12

        rows = batch_size // cols + 1

        for i in range(images_from_batch):

            image = images[i].permute(1, 2, 0).numpy() * std + mean   # Размерность RGB в конец

            plt.subplot(rows, cols, i+1)

            plt.xticks([])

            plt.yticks([])

            plt.grid(False)

            plt.imshow(image.clip(0, 1))

            if classes is not None:

                plt.xlabel(classes[labels[i].numpy()])

        plt.show()

        

        break
batch_size = 500



train_dir = join(root_dir, 'train')

val_dir = join(root_dir, 'val')



classes = ['No', 'Cactus']



train_transforms1 = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



train_transforms2 = transforms.Compose([

    transforms.RandomHorizontalFlip(p=1),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



train_transforms3 = transforms.Compose([

    transforms.RandomVerticalFlip(p=1),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



train_transforms4 = transforms.Compose([

    transforms.RandomHorizontalFlip(p=1),

    transforms.RandomVerticalFlip(p=1),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



val_transforms = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])



#train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)

tds1 = torchvision.datasets.ImageFolder(train_dir, train_transforms1)

tds2 = torchvision.datasets.ImageFolder(train_dir, train_transforms2)

tds3 = torchvision.datasets.ImageFolder(train_dir, train_transforms3)

tds4 = torchvision.datasets.ImageFolder(train_dir, train_transforms4)



train_dataset = torch.utils.data.ConcatDataset([tds1, tds2, tds3, tds4])



val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)



train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)





for images, labels in train_dataloader:

    print(images.size())

    print(labels.size())

    break
show_sample_images(train_dataloader, batch_size, 72, denormalize=True)
print(f'Batch size: {batch_size}')

print(f'Train batches: {len(train_dataloader)}, Train samples: {len(train_dataset)}')

print(f'Val batches:   {len(val_dataloader)}, Val samples:    {len(val_dataset)}')
train_batch_loss_history = []

train_batch_accuracy_history = []



train_loss_history = []

train_accuracy_history = []



val_loss_history = []

val_accuracy_history = []



def validate(model, loss, optimizer):

        

    dataloader = val_dataloader

    model.eval()   # Set model to evaluate mode



    sum_loss = 0.

    sum_accuracy = 0.



    for inputs, labels in dataloader:

        inputs = inputs.cuda()

        labels = labels.cuda()



        optimizer.zero_grad()



        with torch.set_grad_enabled(False):

            preds = model(inputs)

            loss_value = loss(preds, labels)

            preds_class = preds.argmax(dim=1)



        sum_loss += loss_value.item()

        sum_accuracy += (preds_class == labels.data).float().mean().cpu().numpy().item()



    val_loss = sum_loss / len(dataloader)

    val_accuracy = sum_accuracy / len(dataloader)



    val_loss_history.append(val_loss)

    val_accuracy_history.append(val_accuracy)

    

    print(f'Validation accuracy {val_accuracy * 100:.2f} %, loss {val_loss:.4f}')



    model.train()  # Вернули как было





def train_model(model, loss, optimizer, scheduler, num_epochs):

        

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}/{num_epochs-1}: ', end='')



        dataloader = train_dataloader

        model.train()  # Set model to training mode



        sum_loss = 0.

        sum_accuracy = 0.



        # Прогон по батчам

        for inputs, labels in dataloader:   #tqdm(dataloader):

            inputs = inputs.cuda()

            labels = labels.cuda()



            optimizer.zero_grad()



            # forward and backward

            with torch.set_grad_enabled(True):

                preds = model(inputs)

                loss_value = loss(preds, labels)

                preds_class = preds.argmax(dim=1)



                loss_value.backward()

                optimizer.step()

                # scheduler.step()



            batch_loss = loss_value.item()

            batch_accuracy = (preds_class == labels.data).float().mean().cpu().numpy().item()



            sum_loss += batch_loss

            sum_accuracy += batch_accuracy

            

            train_batch_loss_history.append(batch_loss)

            train_batch_accuracy_history.append(batch_accuracy)

            #print(f'\r----- {phase}, batch accuracy {train_batch_accuracy * 100:.2f} %, batch loss {train_batch_loss:.4f}')        

            #validate(model, loss, optimizer)

            

        epoch_loss = sum_loss / len(dataloader)

        epoch_acc = sum_accuracy / len(dataloader)



        train_loss_history.append(epoch_loss)

        train_accuracy_history.append(epoch_acc)

        scheduler.step()



        # Валидация

        # print('\n End epoch: ', end='')

        validate(model, loss, optimizer)

        

    return model
model = models.resnet50(pretrained=True)

#model = models.mobilenet_v2(pretrained=True)



# for param in model.parameters():

#     param.requires_grad = False



model.fc = torch.nn.Linear(model.fc.in_features, 2)

#model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)



model = model.cuda()



loss = torch.nn.CrossEntropyLoss() #weight=torch.FloatTensor([1, 1]).cuda())

optimizer = torch.optim.Adam(model.parameters())#, lr=1.0e-3, weight_decay=0.01, amsgrad=True)



scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.33)

#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)  
print(f'Batch size: {batch_size}\nBatches: {len(train_dataloader)}\nAll elements: {len(train_dataset)}')
epochs = 7



train_model(model, loss, optimizer, scheduler, num_epochs=epochs);
plt.figure(figsize=(20,10))

    

plt.subplot(1, 3, 1)

plt.plot(train_batch_loss_history, label='Train Batch Loss')

plt.plot(train_batch_accuracy_history, label='Train Batch Accuracy')

plt.legend();



plt.subplot(1, 3, 2)

plt.plot(train_accuracy_history, label='Train accuracy')

plt.plot(val_accuracy_history, label='Val accuracy')

plt.legend();

    

plt.subplot(1, 3, 3)

plt.plot(train_loss_history, label='Train Loss')

plt.plot(val_loss_history, label='Val Loss')

plt.legend();
os.makedirs(join(root_dir, 'test'), exist_ok=True)



test_dir = join(root_dir, 'test')



os.makedirs(join(test_dir, 'unknown'), exist_ok=True)



source_dir = join(input_dir, 'test', 'test')



for i, filename in enumerate(tqdm(sorted(os.listdir(source_dir)))):

    shutil.copy(join(source_dir, filename), join(test_dir, 'unknown', filename))

    

    if i < 10:

        print(filename)
test_transforms = val_transforms



test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)



test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)



for images, labels in test_dataloader:

    print(images.size())

    print(labels.size())

    break
show_sample_images(test_dataloader, batch_size, 12, denormalize=True, classes=['Unknown'])
model.eval()



test_predictions = []



i = 1

for images, labels in test_dataloader:

    images = images.cuda()

    with torch.set_grad_enabled(False):

        preds = model(images)

    test_predictions.append(T.softmax(preds, dim=1)[:,1].data.cpu().numpy())

    print(f'\r{i}/{len(test_dataloader)}', end='')

    i += 1

    

test_predictions = np.concatenate(test_predictions)  

test_predictions = (test_predictions >= 0.5).astype('int')
test_files = next(iter(os.walk(join(test_dir, 'unknown'))))[2]



print(sorted(test_files)[:10])
submission_df = pd.DataFrame.from_dict({'id': sorted(test_files), 'has_cactus': test_predictions})

submission_df.set_index('id', inplace=True)

submission_df.head(25)
submission_df.to_csv('submission.csv')
!rm -rf train val test