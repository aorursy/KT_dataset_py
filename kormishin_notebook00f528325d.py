import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



print(os.listdir("../input/tiny-imagenet/tiny-imagenet-200"))
import torch

from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Normalize, RandomHorizontalFlip, RandomErasing, RandomPerspective, RandomRotation



composed_transformers = Compose([

    ToTensor(),

    # Аугментация

    #RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3, fill=0),

    #RandomRotation(15, resample=False, expand=False, center=None, fill=None)

    #RandomHorizontalFlip(p=0.5),

    #RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

    #Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225)),    

])

dataset = ImageFolder('../input/tiny-imagenet/tiny-imagenet-200/train', composed_transformers)



# делим выборку на train, validation (cv_data) и test (val_data)

torch.manual_seed(42)

train_data, cv_data, val_data = torch.utils.data.random_split(dataset, [80000, 10000, 10000])

#val_data = ImageFolder('../input/tiny-imagenet/tiny-imagenet-200/test', composed_transformers)

#test_data = ImageFolder('../input/tiny-imagenet/tiny-imagenet-200/val', composed_transformers)

len(train_data), len(cv_data), len(val_data)
# загрузчики выборок



# размер батча

batch_size = 256



# train выборка

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)



# validation выборка для проверки train

cv_loader = torch.utils.data.DataLoader(cv_data, batch_size=batch_size, shuffle=False, num_workers=4)



# тестовая выборка для проверки финального accuracy

val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
# выведем батч с картинками (16 шт.) для выборки train

from torchvision.utils import make_grid



for images, _ in train_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))

    break
# выведем батч с картинками (16 шт.) для выборки валидационной выборки (cv_data)

from torchvision.utils import make_grid



for images, _ in cv_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))

    break
# выведем батч с картинками (16 шт.) для выборки Тестовой выборки (val_data)

from torchvision.utils import make_grid



for images, _ in val_loader:

    print('images.shape:', images.shape)

    plt.figure(figsize=(16,8))

    plt.axis('off')

    plt.imshow(make_grid(images, nrow=8).permute((1, 2, 0)))

    break
# доступность GPU

# если мы на GPU, то будет выведено соответствующее уведомление, иначе - отобразится CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if(not torch.cuda.is_available()):

    print('Текущий device - CPU. Необходимо включить GPU (видеокарта не подключена)')

else:

    print(device)
import torch.nn as nn

import torch.nn.functional as F



class Flatten(nn.Module):

    def forward(self, input):

        return input.view(input.size(0), -1)
class CNNClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        #self.device = device

        #self.conv0 = nn.Conv2d(3, 32, kernel_size=3)

        #self.maxpool0 = nn.MaxPool2d((2,2))

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

        #self.bnorm1 = nn.BatchNorm2d(64)

        #self.drop1 = nn.Dropout(p=0.3)

        self.maxpool1 = nn.MaxPool2d((4,4))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)

        self.maxpool2 = nn.MaxPool2d((2,2))

        #self.conv3 = nn.Conv2d(64, 3, kernel_size=3)

        #self.bnorm2 = nn.BatchNorm2d(3)

        #self.drop2 = nn.Dropout(p=0.5)

        self.flatten = Flatten()

        # kernel = 3,  conv1-64, maxpool 4x4

        self.fc1 = nn.Linear(4608, 1024)

        self.fc2 = nn.Linear(1024, 200)

        

        #self.fc1 = nn.Linear(512, 256)

        #self.fc2 = nn.Linear(256, 200)

        # kernel = 3,  conv1-64, maxpool 4x4

        #self.fc1 = nn.Linear(14400, 4096)

        #self.fc2 = nn.Linear(4096, 200)

        # kernel = 3, conv-16 x 3 - maxpool 4x4

        #self.fc = n.Linear(3600, 200)

        # kernel = 3, conv-16 x 3 - maxpool 2x2

        #self.fc = nn.Linear(15376, 200)

        # maxpooling, kernel = 3, conv-64, maxpool 4x4

        #self.fc = nn.Linear(14400, 200)

        # без maxpooling, kernel = 3, conv-64 x 3

        #self.fc = nn.Linear(10092, 200)

        # без maxpooling, kernel = 3

        # self.fc = nn.Linear(10800, 200)

        # для maxpooling, kernel = 3

        #self.fc = nn.Linear(972, 200)

        # без maxpooling, kernel = 2

        #self.fc = nn.Linear(11532, 200)

        

    

    def forward(self, x):

        # forward pass сети

        # умножение на матрицу весов 1 слоя и применение функции активации

        #x = F.relu(self.conv0(x))

        #x = self.maxpool0(x)

        x = F.relu(self.conv1(x))

        #x = self.bnorm1(x)

        #x = self.drop1(x)

        x = self.maxpool1(x)

        x = F.relu(self.conv2(x))

        x = self.maxpool2(x)

        #x = F.relu(self.conv3(x))

        #x = self.bnorm2(x)

        #x = self.drop2(x)

        x = self.flatten(x)

        # умножение на матрицу весов 2 слоя и применение функции активации

        #print(x.shape)

        #x = F.softmax(self.fc(x))

        #x = F.relu(self.fc(x))

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x



cnn_model = CNNClassifier()



# переносим сеть на GPU

cnn_model.to(device)
# Loss-функция и оптимизатор



from torch import optim

from torch.optim.lr_scheduler import CosineAnnealingLR



LEARNING_RATE = 7e-4



criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(

    filter(lambda p: p.requires_grad, cnn_model.parameters()),

    lr=LEARNING_RATE,

)



scheduler = CosineAnnealingLR(optimizer, 1)
# определяем функцию и необходимые шаги для тренировки



def train_epoch(model, optimizer, train_loader):

    model.train()

    total_loss, total = 0, 0

    for i, (inputs, target) in enumerate(train_loader):

        # переносим на GPU

        inputs = inputs.to(device)

        target = target.to(device)

        

        # Reset gradient

        optimizer.zero_grad()

        

        # Forward pass

        output = model(inputs)

        

        # Compute loss

        loss = criterion(output, target)

        

        # Perform gradient descent, backwards pass

        loss.backward()



        # Take a step in the right direction

        optimizer.step()

        scheduler.step()



        # Record metrics

        total_loss += loss.item()

        total += len(target)



    return total_loss / total



# функция и шаги для валидации

def validate_epoch(model, cv_loader):

    model.eval()

    total_loss, total = 0, 0

    with torch.no_grad():

        for inputs, target in cv_loader:           

            # переносим на GPU

            inputs = inputs.to(device)

            target = target.to(device)

            

            # Forward pass

            output = model(inputs)



            # Calculate how wrong the model is

            loss = criterion(output, target)



            # Record metrics

            total_loss += loss.item()

            total += len(target)



    return total_loss / total
# тренируем

from tqdm import tqdm



max_epochs = 45

n_epochs = 0

train_losses = []

valid_losses = []



for epoch_num in range(max_epochs):



    train_loss = train_epoch(cnn_model, optimizer, train_loader)

    valid_loss = validate_epoch(cnn_model, val_loader)

    

    tqdm.write(

        f'эпоха #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.2e}\tvalid_loss: {valid_loss:.2e}\n',

    )

    

    # Early stopping (ранняя остановка) если текущий valid_loss (значение функции ошибки для валидаионной выборки) больше чем 10 последних valid losses

    if len(valid_losses) > 4 and all(valid_loss >= loss for loss in valid_losses[-5:]):

        print('Stopping early')

        break

    

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

    

    n_epochs += 1



print('Тренировка окончена')
# строим график loss-функции после тренировки

epoch_ticks = range(1, n_epochs + 1)

plt.plot(epoch_ticks, train_losses)

plt.plot(epoch_ticks, valid_losses)

plt.legend(['Train Loss', 'Valid Loss'])

#plt.legend(['Train Loss'])

plt.title('Losses') 

plt.xlabel('Epoch #')

plt.ylabel('Loss')

plt.xticks(epoch_ticks)

plt.show()
# Эффективность CNN-модели

from sklearn.metrics import accuracy_score



cnn_model.eval()

test_accuracy, n_examples = 0, 0

y_true, y_pred = [], []



with torch.no_grad():

    for inputs, target in val_loader:

       

        inputs = inputs.to(device)

        target = target.to(device)

        

        probs = cnn_model(inputs)

        

        probs = probs.detach().cpu().numpy()

        predictions = np.argmax(probs, axis=1)

        target = target.cpu().numpy()

        

        y_true.extend(predictions)

        y_pred.extend(target)

        

test_accuracy = accuracy_score(y_true, y_pred)



print("Final results:")

print("  test accuracy:\t\t{:.2f} %".format(

    test_accuracy * 100))



if test_accuracy * 100 > 40:

    print("Achievement unlocked: 110lvl Warlock!")

elif test_accuracy * 100 > 35:

    print("Achievement unlocked: 80lvl Warlock!")

elif test_accuracy * 100 > 30:

    print("Achievement unlocked: 70lvl Warlock!")

elif test_accuracy * 100 > 25:

    print("Achievement unlocked: 60lvl Warlock!")

else:

    print("We need more magic! Follow instructons below")
