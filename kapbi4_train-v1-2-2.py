#!/usr/bin/python3

# Импорт библиотек



import numpy as np

#import matplotlib.pyplot as plt



import os

import torch

from torch import nn, optim

from torchvision import datasets, transforms, models

from torch.utils.data.sampler import SubsetRandomSampler

from tqdm import tqdm



# на вход подается изображения размером 224 на 224



v = 'v1.2.2'

n_class = 8 # количество выходных классов

freeze = 0 # есть возможность заморозить слои

lr = 0.0001 # lerning rate

n_epoch = 500 # количество эпох

patience = 15

batch_size = 64



# используем графический процессор если есть

device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

#device = torch.device('cpu')



# Получить текущий lr

def get_lr(optimizer):

    for param_group in optimizer.param_groups:

        return param_group['lr']



# Загрузить тренировочные данные и тестовые данные

def loadset():

    img_size = 224

    

    train_transform = transforms.Compose([

        transforms.Resize(img_size, interpolation=2),

        #transforms.RandomSizedCrop(img_size),

        #transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),

        #transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(degrees=(-7,7), expand=True),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])])



    test_transform = transforms.Compose([

        transforms.Resize(img_size, interpolation=2),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])])



    trainset = datasets.ImageFolder(root='./dataset',

        transform=train_transform)



    testset = datasets.ImageFolder(root='./dataset',

        transform=test_transform)



    num_train = len(trainset)

    indices = list(range(num_train))

    split = int(np.floor(0.1 * num_train))

    np.random.shuffle(indices)



    train_idx, test_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)

    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(trainset,

                                              #shuffle=True,

                                              sampler=train_sampler,

                                              batch_size=batch_size)

    testloader = torch.utils.data.DataLoader(testset,

                                             #shuffle=True,

                                             sampler=test_sampler,

                                             batch_size=batch_size)

    return trainloader, testloader, len(train_idx), len(test_idx)



trainloader, testloader, ltrn, ltst = loadset()



# Показать количество сэмплов в датасете

print(ltrn)

print(ltst)



model_name = 'ResNet50'

model_conv=models.resnet50(pretrained=True)

#model_conv = torch.load('model.pth')



# Заменяем последний выходной слой

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, n_class)



# Показать модель

for name, child in model_conv.named_children():

    for name2, params in child.named_parameters():

        print(name, name2)



# Фризим первые "freeze" слоев

# conv1, bn1, layer1, layer2, layer3, layer4, fc

#ct = 0

#for name, child in model_conv.named_children():

#    ct += 1

#    if ct < freeze: # 7

#        for name2, params in child.named_parameters():

#            params.requires_grad = False



model_conv = model_conv.to(device)

#optimiser = optim.SGD(model_conv.parameters(), lr=lr, momentum=0.85)

#optimiser = optim.SGD(model_conv.parameters(), lr=lr, momentum=0.9)

optimiser = optim.Adam(model_conv.parameters(), lr=lr)

criterion = nn.CrossEntropyLoss()

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.8, patience=2)



train_loss = []

train_losses = []

test_acc = []

train_elosses = []

val_loss = []

valid_loss_min = np.Inf

p = 0

stop = False

for epoch in range(n_epoch):



    # обучаем модель

    model_conv.train()

    train_eloss = []

    #for i, (data, target) in enumerate(trainloader):

    for data, target in tqdm(trainloader):

        data, target = data.to(device=device), target.to(device=device)

        optimiser.zero_grad()

        output = model_conv(data)



        loss = criterion(output, target)

        train_loss.append(loss.item())

        train_eloss.append(loss.item())



        loss.backward()

        optimiser.step()



    # тренируем модель

    model_conv.eval()

    val_eloss = []

    correct = 0

    with torch.no_grad():

        for data, target in tqdm(testloader):

            data, target = data.to(device=device), target.to(device=device)

            output = model_conv(data)



            loss = criterion(output, target)

            val_loss.append(loss.item())

            val_eloss.append(loss.item())



            _, predicted = torch.max(output.data, 1)

            correct += (predicted == target).sum().item()



    acc = 1 - correct / ltst

    test_acc = [acc]



    valid_loss = np.mean(val_loss)

    valid_eloss = np.mean(val_eloss)

    train_losses.append(np.mean(train_loss))

    train_elosses.append(np.mean(train_eloss))



    print("epoh: " + str(epoch), "lr: " + str(get_lr(optimiser)))

    print("train_loss: ", np.mean(train_loss))

    print("train_eloss: ", np.mean(train_eloss))

    print("val_loss: ", valid_loss)

    print("val_eloss: ", valid_eloss)

    print('acc: ', acc)



    # опираясь на последнюю эпоху

    # scheduler.step(valid_eloss)

    # if valid_eloss <= valid_loss_min:

    #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

    #                    valid_loss_min,

    #                    valid_eloss))

    #    torch.save(model_conv, 'model.pth')

    #    valid_loss_min = valid_eloss

    #    p = 0



    # опираясь на все эпохи

    scheduler.step(valid_loss)

    if valid_loss <= valid_loss_min:

        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(

                        valid_loss_min,

                        valid_loss))

        torch.save(model_conv, 'model.pth')

        valid_loss_min = valid_loss

        p = 0





    # проверяем как дела на валидации

    # опираясь на последнюю эпоху

    # if valid_eloss > valid_loss_min:

    # опираясь на все эпохи

    if valid_loss > valid_loss_min:

        p += 1

        print(f'{p} epochs of increasing val loss')

        if p > patience:

            print('Stopping training')

            stop = True

            break



    # записываем для истории ;)

    # torch.save(model_conv, 'acc_{}-loss_{}-model.pth'.format(

    #    acc,

    #    np.mean(val_eloss)))

    # можно сохранять все состояния

    # state = {

    #     'epoch': epoch,

    #     'state_dict': model.state_dict(),

    #     'optimizer': optimizer.state_dict(),

    #     ...

    # }

    # torch.save(state, filepath)



    # а так же их загрузить для продолжения обучения

    # model.load_state_dict(state['state_dict'])

    # optimizer.load_state_dict(state['optimizer'])



    if stop:

        break



exit()



#!/usr/bin/python3

# Импорт библиотек



import torch

import os

from torchvision import datasets, transforms, models

from tqdm import tqdm



# на вход подается изображения размером 224 на 224



# Загружаем натренорованную модель

filemodel = "model.pth"

model_conv = torch.load(filemodel, map_location='cpu')



PATH = './'

test_path = os.path.join(PATH, "test/")



batch_size = 64



f = open('submission.csv', 'w')

f.write('image_name,label'+'\r\n')



# используем графический процессор если есть

device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')

#device = torch.device('cpu')



# Загрузить валидационные данные

def loadset():

    test_transform = transforms.Compose([

        transforms.Resize(224, interpolation=2),

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])])



    testset = datasets.ImageFolder(root=test_path,

        transform=test_transform)



    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 

        shuffle=False)

    

    return testloader, len(testset)



testloader, ltst = loadset()



# Показать количество сэмплов в датасете

print(ltst)



model_conv = model_conv.to(device)

model_conv.eval()



count = 0

val_loss = []

with torch.no_grad():

    for i, (data, _) in enumerate(tqdm(testloader), 0):

        data = data.to(device=device)

        outputs = model_conv(data)



        _, predicted = torch.max(outputs.data, 1)

        sample_fname, _ = testloader.dataset.samples[i+1]

        items = testloader.dataset.samples[count:len(data)]



        # Сохраняем результат предсказания

        for j, (p) in enumerate(predicted, 0):

            filename, _ = testloader.dataset.samples[count:count+len(data)][j]

            filename = filename.split('/')[-1]

            filename = filename.split('.')[0] + "." + filename.split('.')[1]

            out =  "{},{}\r\n".format(filename, p.item())

            f.write(out)

        count += len(predicted)



f.close

exit()


