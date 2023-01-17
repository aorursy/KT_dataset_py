import os

from PIL import Image

import matplotlib

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook 

import copy

import pickle

import pandas as pd



import torch

import torchvision

from torch.utils.data import DataLoader, Dataset, random_split

import torchvision.transforms as transforms

from torchvision import datasets



import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



import numpy as np

import pandas as pd

from sklearn import preprocessing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
def label(source_path):

  file_name = []

  label = []

  filedir=[root+'/'+file for root,dirs,files in os.walk(source_path) for file in files]

  for count,file in enumerate(filedir[:100000]):

      if (file[-5]) == '1':

          file_name.append(file)

          label.append(1)

      else:

          file_name.append(file)

          label.append(0)

  

  df = pd.DataFrame({'name':file_name})

  df1=pd.DataFrame({'label':label})

  df=pd.concat([df,df1], axis=1)

  #print(df.groupby('label')['label'].count())

  return df
label('/kaggle/input')
# Dataloader
#For converting the dataset to torchvision dataset format

class CancerDataset(Dataset):

    def __init__(self, panda, transform=None):

        self.transform = transform

        self.panda = panda

        y=self.panda['name']

        self.file_names=list(y)

        self.len = len(self.file_names)

        if self.panda is not None:

            self.classes_mapping, self.classes_encoding = self.get_classes(panda)

            

    def __len__(self):

        return len(self.file_names)

    

    def __getitem__(self, index):

        file_name = self.file_names[index]

        image_data = self.pil_loader(file_name).resize((50,50)) 

        if self.transform:

            image_data = self.transform(image_data)

                  

        row = self.classes_mapping.iloc[:,:]  

        class_, class_code = int(row.iloc[[index],0]), int(row.iloc[[index],1])

        return image_data, class_, class_code

          

    def pil_loader(self,path):

        with open(path, 'rb') as f:

            img = Image.open(f)

            return img.convert('RGB')

      

    def get_classes(self, panda):

        classes_mapping = pd.DataFrame(panda).iloc[:,1:]

        le = preprocessing.LabelEncoder()

        classes_mapping['Class_code'] = le.fit_transform(classes_mapping['label'])

        classes_encoding = {}

        for code, class_ in enumerate(list(le.classes_)):

            classes_encoding[code] = class_

        return classes_mapping, classes_encoding
transform = transforms.Compose([ 

                transforms.ToTensor()

])

#,transforms.Normalize((0.5,), (0.5,))
full_data = CancerDataset(label('/kaggle/input'), transform)



full_data.classes_encoding
train_size = int(0.8 * len(full_data))

test_size = len(full_data) - train_size

train_data, validation_data = torch.utils.data.random_split(full_data, [train_size, test_size])



print(len(full_data), len(train_data), len(validation_data))
# Data loaders for training and validation datasets. Dataloaders provide shuffled data in batches

train_loader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=20, shuffle=False)

dataiter = iter(train_loader)

images,_, labels = dataiter.next()
class BreastcancerCNN(nn.Module):

    def __init__(self, p=0):

        super(BreastcancerCNN, self).__init__()

        self.p = p

        self.features = nn.Sequential(

            # layer 1 (3 , 50, 50 ) to (64, 25, 25)

            nn.Conv2d(3, 64, 4,2,1),  # 41 x 85 -> 21 x 43

            nn.BatchNorm2d(64),

            nn.ReLU(),

            #pooling (64 , 50, 50 ) to (64, 25, 25)

            #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 11 x 22 -> 6 x 12



            # layer 2   (64 , 25, 25 ) to (128, 12, 12)         

            nn.Conv2d(64, 128,3,2,0),  # 21 x 43 -> 11 x 22

            nn.BatchNorm2d(128),

            nn.ReLU(),  

            #pooling (128 , 12, 12 ) to (128, 6, 6)

            nn.MaxPool2d(2,2,0),  # 11 x 22 -> 6 x 12

            #nn.Dropout2d(p=self.p),



            # layer 3 (128 , 6, 6 ) to (256, 4, 4)

            nn.Conv2d(128, 256, 2,2,1),  # 6 x 12 -> 4 x 7

            nn.BatchNorm2d(256),

            nn.ReLU()         

        )



        self.fc = nn.Sequential(

            # Fully connected layer            

            nn.Linear(256*4*4, 512),

            nn.Sigmoid(),

            #nn.Dropout(p=self.p),



            # Classifier layer

            nn.Linear(512, 2),

            nn.Sigmoid()

        )



    def forward(self, x):

        x = self.features(x)

        #print(x.shape)

        x = x.view(x.size(0), -1)

        #print(x.shape)

        x = self.fc(x)

        return x
def train(model, loss_fn, opt, epoch, batch_log_interval, dropout_prob):

    

    model.train()

    model.p = dropout_prob

    loss_avg = 0

    

    for batch_id, data in enumerate(train_loader):

        inputs, _, labels = data

        inputs = inputs.to(device)

        labels = labels.to(device)

        #print(labels)

        

        opt.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs,labels)        

        loss_avg += loss.data.item()*len(inputs) # as 'loss' is avg of total mini-batch loss

        loss.backward()

        opt.step()



        '''if batch_id % batch_log_interval == 0:            

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg batch loss: {:.3f}'.format(

                epoch, batch_id * len(inputs), len(train_loader.dataset),

                100. * batch_id / len(train_loader), loss.data.item()/len(inputs)))'''

            

        del inputs, labels, outputs

        torch.cuda.empty_cache()

        

    loss_avg /= len(train_loader.dataset)

    print('\nEpoch: {}, Train set: Average loss: {:.4f}'.format(epoch, loss_avg)) 

    return loss_avg
def validate(model, loss_fn, opt, epoch):    

    model.eval()

    model.p = 0

    validation_loss = 0

    correct = 0



    with torch.no_grad():

        for inputs, _, labels in validation_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

               

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            validation_loss += loss.data.item()*len(inputs)

            output_pred = outputs.data.max(1)[1]

            correct += output_pred.eq(labels).sum().item()

          

        validation_loss /= len(validation_loader.dataset)

        validation_accuracy = 100.0 * correct / len(validation_loader.dataset)

        print('\nEpoch: {}, Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.

              format(epoch, validation_loss, correct, len(validation_loader.dataset), validation_accuracy))  

        

        return validation_loss, validation_accuracy
def main(lr, momentum, best_model=None, max_validation_accuracy=0, dropout_prob=0):    

    train_loss = []

    validation_loss = []

    validation_accuracy = []



    print('\nLR = {}, Momentum = {}\n'.format(lr, momentum))

    torch.manual_seed(0)

    model = BreastcancerCNN(dropout_prob)

    if best_model:

        model.load_state_dict(best_model)

    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    opt = optim.SGD(model.parameters(), lr=lr, momentum = momentum)    



    for epoch in tqdm_notebook(range(1, epochs + 1), total=epochs, unit="epoch"):

        train_loss_epoch = train(model,loss_fn, opt, epoch, 1, dropout_prob)             

        if epoch % 1 == 0:

            train_loss.append(train_loss_epoch)

            valid_loss_epoch, valid_accuracy_epoch = validate(model,loss_fn, opt, epoch)

            validation_loss.append(valid_loss_epoch)

            validation_accuracy.append(valid_accuracy_epoch)



            if valid_accuracy_epoch > max_validation_accuracy:

                max_validation_accuracy = valid_accuracy_epoch

                best_model = copy.deepcopy(model.state_dict())

                best_epoch = epoch

            print("Maximum validation accuracy so far: {:.0f}%".format(max_validation_accuracy))

        print("-----------------------------------------------------------------\n")



    return best_model, best_epoch, train_loss, validation_loss, validation_accuracy
lr = 0.001

momentum = 0.9

epochs = 30
#model = torch.load('models/model_e1.pt')

best_model, best_epoch, train_loss, validation_loss, validation_accuracy = main(lr, momentum)
!mkdir models/



# model_e2 means lr=1e-2, similar names will be used

torch.save(best_model, 'models/model.pt')
def plot_loss(epochs, metric, lr, batch_size, data_type, metric_type):

    plt.plot(epochs,metric)

    plt.title(" {} {}, lr={}, batch size={}".format(data_type, metric_type, lr, batch_size))

    plt.xlabel("epochs")

    if metric_type == "loss":

        ylabel = "Avergae CE loss"

    else:

        ylabel = "Accuracy (%)"

    plt.ylabel("Avergae CE loss")  

    plt.show()
lr = 0.01

plot_loss(range(1,31), validation_accuracy, lr, 20, "validation", "accuracy")



plt.plot(range(1,31), train_loss, c='r', label="train")

plt.plot(range(1,31), validation_loss, c='g', label="validation")

plt.title("Drop out (p=0.5), lr=0.01(60 epochs)/0.001(30 epochs), batch size={}".format(20))

plt.xlabel("epochs")

plt.ylabel("Avergae CE loss")

plt.legend(loc='upper right')

plt.show()