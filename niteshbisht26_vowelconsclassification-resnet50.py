import os

from PIL import Image

import matplotlib.pyplot as plt



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms



#For converting the dataset to torchvision dataset format

class VowelConsonantDataset(Dataset):

    def __init__(self, file_path,train=True,transform=None):

        self.transform = transform

        self.file_path=file_path

        self.train=train

        self.file_names=[file for _,_,files in os.walk(self.file_path) for file in files]

        self.len = len(self.file_names)

        if self.train:

            self.classes_mapping=self.get_classes()

    

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name=self.file_names[index]

        image_data=self.pil_loader(self.file_path+"/"+file_name)

        if self.transform:

            image_data = self.transform(image_data)

        if self.train:

            file_name_splitted=file_name.split("_")

            Y1 = self.classes_mapping[file_name_splitted[0]]  # Y1 gets vowels     

            Y2 = self.classes_mapping[file_name_splitted[1]]  # Y2 gets consonants

            z1,z2=torch.zeros(10),torch.zeros(10)

            z1[Y1-10],z2[Y2]=1,1

            label=torch.stack([z1,z2])    # z1 -> vowels   z2 -> consonants



            return image_data, label



        else:

            return image_data, file_name

          

    def pil_loader(self,path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            return img.convert('RGB')



      

    def get_classes(self):

        classes=[]

        for name in self.file_names:

            name_splitted=name.split("_")

            classes.extend([name_splitted[0],name_splitted[1]])

        classes=list(set(classes))

        classes_mapping={}

        for i,cl in enumerate(sorted(classes)):

            classes_mapping[cl]=i

        return classes_mapping
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import torchvision

import matplotlib.pyplot as plt

from torchvision import datasets, models



import torchvision.transforms as transforms



import numpy as np

import pandas as pd



from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
transform = transforms.Compose([

    transforms.RandomResizedCrop(64),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize((0.5074, 0.4633, 0.4228), (0.3233, 0.3129, 0.3335))])



batch_size = 128
full_data = VowelConsonantDataset("../input/padhai-hindi-vowel-consonant-classification/train/train/",

                                  train=True, transform=transform)

train_size = int(0.9 * len(full_data))

test_size = len(full_data) - train_size



train_data, validation_data = random_split(full_data, [train_size, test_size])



train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True)
test_data = VowelConsonantDataset("../input/padhai-hindi-vowel-consonant-classification/test/test/",

                                  train=False,transform=transform)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
x, y = next(iter(train_loader))

print(x.shape, y.shape)
def imshow(img, label):

    label = label.squeeze()

    title = f'V_{np.argmax(label[0])}C_{np.argmax(label[1])}'

    npimg = img.numpy()

    npimg = np.transpose(npimg, (1, 2, 0))

    plt.imshow(npimg)

    plt.title(title)

    plt.axis('off')

    plt.show()
images, labels = next(iter(train_loader))

temp = torchvision.utils.make_grid(images)

print(temp.shape)

imshow(temp, labels)
in_features = x.shape[0] * x.shape[1] * x.shape[2]

print(in_features)
class MyNet(nn.Module):

    def __init__(self):

        super(MyNet, self).__init__()

        self.resnet50 = nn.Sequential(*list(model.children())[:-1])

        self.vow_classifier = nn.Linear(2048, 10)

        self.cons_classifier = nn.Linear(2048, 10)

        

    def forward(self, x):

        aux_out = self.resnet50(x)

        aux_out = aux_out.squeeze()

        return self.vow_classifier(aux_out), self.cons_classifier(aux_out)
def get_labels(Y_train):

    vow_labels = []

    cons_labels = []

    for i in range(Y_train.shape[0]):

        for j in [0,1]:

            if j == 0:

                vow_labels.append(Y_train[i][j].unsqueeze(0))

            else:

                cons_labels.append(Y_train[i][j].unsqueeze(0))

    vow_labels = torch.argmax(torch.cat(vow_labels, axis=0), axis=1)

    cons_labels = torch.argmax(torch.cat(cons_labels, axis=0), axis=1)

    return vow_labels, cons_labels
def evaluate(dataloader, model_state=None):

    total_vow, correct_vow = 0, 0

    total_cons, correct_cons = 0, 0

    total_acc = 0

    

    if model_state is None:

        custom_model.load_state_dict(checkpoint['state'])

    else:

        custom_model.load_state_dict(model_state)

    

    custom_model.eval()

    with torch.no_grad():

        for X, Y in dataloader:

            vow_labels, cons_labels = get_labels(Y)

            X, vow_labels, cons_labels = X.to(device), vow_labels.to(device), cons_labels.to(device)

            

            y_hat_vow, y_hat_cons = custom_model(X)

            total_cons += cons_labels.shape[0]

            total_vow += vow_labels.shape[0]

            

            y_hat_vow = torch.argmax(y_hat_vow, axis=1)

            y_hat_cons = torch.argmax(y_hat_cons, axis=1)

            

            correct_vow += (y_hat_vow == vow_labels).sum().item()

            correct_cons += (y_hat_cons == cons_labels).sum().item()

            

            total_acc += ((y_hat_vow == vow_labels) == (y_hat_cons == cons_labels)).sum().item()

    

    acc_vow = 100 * correct_vow/total_vow

    acc_cons = 100 * correct_cons/total_cons

    total_acc = 100 * total_acc/total_cons

    

    return acc_vow, acc_cons, total_acc
loss_arr = []

checkpoint = {}

def fit(epochs=10, lr=0.0001, tb=None):

    opt = optim.Adam(custom_model.parameters(), lr=lr, weight_decay=0.01)

#     scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    loss_fn = nn.CrossEntropyLoss()

    min_loss = 5

    

    for epoch in tqdm(range(epochs), total=epochs, unit="epoch"):

        

        for X_train, Y_train in train_loader:

            

            vow_labels, cons_labels = get_labels(Y_train)

            

            X_train, vow_labels, cons_labels = X_train.to(device), vow_labels.to(device), cons_labels.to(device)

            

            y_hat_vow, y_hat_cons = custom_model(X_train)

            

            vow_loss = loss_fn(y_hat_vow, vow_labels)

            cons_loss = loss_fn(y_hat_cons, cons_labels)

            combined_loss = vow_loss + cons_loss

            

            opt.zero_grad()

            combined_loss.backward()

            opt.step()

            

            if combined_loss.item() < min_loss:

                min_loss = combined_loss.item()

                checkpoint['loss'] = (min_loss, epoch)

                checkpoint['state'] = custom_model.state_dict()



#         scheduler.step()

            

#         Logging training and validation accuracy



#         _, _, train_tot = evaluate(train_loader, custom_model.state_dict())

#         _, _, val_tot = evaluate(validation_loader, custom_model.state_dict())

#         tb.add_scalar('Training Accuracy', train_tot, epoch)

#         tb.add_scalar('Validation Acccuracy', val_tot, epoch)

        

        print('Epoch {}/{} Loss = {}'.format(epoch+1, epochs, round(combined_loss.item(), 4)))

        loss_arr.append(combined_loss.item())

        

        # Tensorboard setup

        if tb is not None:

            tb.add_scalar('Loss', combined_loss.item(), epoch)

            for name, param in custom_model.named_parameters():

                tb.add_histogram(name, param, epoch)

                tb.add_histogram(f'{name}.grad', param.grad, epoch)
model = models.resnet50()

model.load_state_dict(torch.load('../input/resnet50/resnet50.pth'))

custom_model = MyNet().to(device)

lr = 0.0001

# train_new_loader = custom_dataloader(input_to_untrained_model, labels, batch_size)

# comment = f'batch_size {batch_size} lr {lr}, Adam)'

# tb = SummaryWriter(comment=comment)

fit(30, lr)

# tb.close()
# rm -r runs
checkpoint['loss']
# library to generate different combinations of hyperparameters

# from itertools import product
small_loss = [loss_arr[i] for i in range(0, len(loss_arr), 5)]

plt.plot(loss_arr)
# Training Accuracy

vow, cons, tot = evaluate(train_loader)

print('Training Accuracy\n***********************')

print('Vowel Accuracy = {} \nConsonant Accuracy = {} \nTotal Accuracy = {}'.format(vow, cons, tot))
# Validation Accuracy

vow, cons, tot = evaluate(validation_loader)

print('Validation Accuracy\n***********************')

print('Vowel Accuracy = {} \nConsonant Accuracy = {} \nTotal Accuracy = {}'.format(vow, cons, tot))
# %load_ext tensorboard.notebook
# %tensorboard --logdir runs
img, fname = next(iter(test_loader))

print(img.shape)
img_id = []

class_ = []



for img, fname in test_loader:

    img = img.to(device)

    custom_model.load_state_dict(checkpoint['state'])

    y_vow, y_cons = custom_model(img)

    

    y_vow = torch.argmax(y_vow, axis=1)

    y_cons = torch.argmax(y_cons, axis=1)

    

    for i in range(y_cons.shape[0]):

        label = f'V{y_vow[i]}_C{y_cons[i]}'

        img_id.append(fname[i])

        class_.append(label)
submission = pd.DataFrame({'ImageId': img_id, 'Class': class_})
submission.head(10)
im = Image.open('../input/padhai-hindi-vowel-consonant-classification/test/test/2614.png')

im = im.convert('RGB')

plt.imshow(im)
submission.to_csv('submission.csv', index=False)