import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

#Pytorch modules
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader,random_split
from torch.functional import F
from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

from PIL import Image

import os
#Loading the csv'
train_csv = pd.read_csv('/kaggle/input/landmark-recognition-2020/train.csv')
test_csv = pd.read_csv('/kaggle/input/landmark-recognition-2020/sample_submission.csv')

#How many images and how many classes
images_amount_train = len(train_csv)
classes_amount_train = len(train_csv['landmark_id'].unique())

print(f"In the train-dataset there is {images_amount_train} images from {classes_amount_train} different classes")
#Count all the amount of images inside each class
image_pr_class = train_csv['landmark_id'].value_counts()
groups = pd.DataFrame(image_pr_class)

groups_new = groups.reset_index(drop=True)
groups_new.plot(kind='hist',bins=100,figsize=(10,10))
#Count the amount of images inside each class 
class_count = train_csv['landmark_id'].value_counts()
class_under5 = class_count[class_count < 5]
class_between5and10 = class_count[(class_count <= 10) & (class_count >= 5)]

print(f"There are {len(class_under5)} classes with under 5 images and {len(class_between5and10)} classes with between 5 and 10 images")
#Statistics of amounts of samples 
groups.describe()
train_csv = train_csv.loc[:20000,:]

#Select classes with 10 or above images 
#image_pr_class = train_csv['landmark_id'].value_counts()
#class_select = image_pr_class[image_pr_class >= 100].index

#train_csv = train_csv.loc[train_csv.landmark_id.isin(class_select)]

#Reset index back to 0
train_csv.reset_index(drop=True, inplace=True)

#Make new label encoder for the new data selection
label_encoder = LabelEncoder()
label_encoder.fit(train_csv.landmark_id.values)
print('found classes', len(label_encoder.classes_))

train_csv.landmark_id = label_encoder.transform(train_csv.landmark_id)

#Calculate the weigths of the classes
def cal_classWeigth(x):
    return 1-(x/20001)

classes = train_csv.landmark_id.value_counts()
class_weights = classes.apply(cal_classWeigth)
class_weights.sort_index(inplace=True)

class_weights = torch.cuda.FloatTensor(class_weights)

image_pr_class = train_csv['landmark_id'].value_counts()
groups = pd.DataFrame(image_pr_class)
groups.to_csv('./images_pr_class')
print(train_csv.landmark_id.value_counts())
ID = 1816
LEARNINGRATE = 0.01
MOMENTUM = 0.9 #Only used for naming
BATCH_SIZE = 64
NUM_CLASSES =  len(label_encoder.classes_) 
NUM_DATASAMPLES = len(train_csv)
EPOCHS = 10
#Defining the dataset class for how to load the data
class Dataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, image_folder,transform):
        self.csv = csv_file
        self.folder = image_folder
        self.transform = transform

  def __len__(self):
        return len(self.csv)

  def __getitem__(self, index):
        image_id = self.csv.iloc[index].id
        image_path = f"{self.folder}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        # Load data and get label
        X = Image.open(image_path)
        y = self.csv.iloc[index].landmark_id
        X = self.transform(X)
        return X, y
#define the wanted tranformation
transform_train=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
                             ])
transform_val=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
                             ])

#Making the dataset from the csv file
dataset = Dataset(train_csv,
                        r'/kaggle/input/landmark-recognition-2020/train'
                        ,transform_val
                       )

msk = np.random.rand(len(train_csv)) < 0.9
train = train_csv[msk]
val = train_csv[~msk]

val_dataset = Dataset(val,
                        r'/kaggle/input/landmark-recognition-2020/train'
                        ,transform_val
                       )
train_dataset = Dataset(train,
                        r'/kaggle/input/landmark-recognition-2020/train'
                        ,transform_train
                       )
#val_dataset, train_dataset = random_split(dataset = dataset, lengths = [int(len(dataset)*0.1),len(dataset)-int(len(dataset)*0.1)])
import random
sample_iter = iter(train_dataset)

rows = 2
cols = 2
axes=[]
fig=plt.figure(figsize=(10,10))
for a in range(rows*cols):
    b = train_dataset.__getitem__(random.randint(0,len(train_dataset)))
    axes.append( fig.add_subplot(rows, cols, a+1) )
    subplot_title=(f'Class: {b[1]}')
    axes[-1].set_title(subplot_title)  
    plt.imshow(b[0].permute(1, 2, 0))
fig.tight_layout()    
plt.show()
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
print(len(train_dataloader))

val_dataloader = DataLoader(val_dataset, batch_size = 1, shuffle=False, num_workers=0)
print(len(val_dataloader))
#Select ReNet18 as network
#ResNet = models.resnet18()
VGG16 = models.vgg16()
net = VGG16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Resnet

#ResNet18
#Change the last layer to fit the classes output
#net.fc = nn.Linear(in_features=512, out_features=NUM_CLASSES, bias=True)

#Resnet50
#Change the last layer to fit the classes output
#net.fc = nn.Linear(in_features=2048, out_features=NUM_CLASSES, bias=True)

#VGG16
net.classifier[6] = nn.Linear(in_features=4096, out_features=NUM_CLASSES, bias=True)

net.to(device)
print(device)
print(net)

class_weights.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNINGRATE)
TrainLoss = []
ValLoss = []
TrainAcc = []
TrainAvgAcc = []
ValAcc = []
ValAvgAcc = []
#EPOCHS
for epoch in range(10):  # loop over the dataset multiple times
    """
    Training loop
    Training on the training data inside train_dataloader
    """
    #Create variables to keep track of loss
    running_loss = 0.0
    trainrunningloss = 0.0
    
    print(f'Epoch: {epoch+1}')
    
    #Set network to train mode
    net.train()
    
    #Create arrays to save labels inside
    correct_labels = []
    predicted_labels = []
    
    
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        trainrunningloss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        predictions = outputs.max(dim=1)[1]
        
        correct_labels.append(labels.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())
        
        running_loss += loss.item()

        # print every 100 mini-batches
        if i % 10 == 9:    
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
        
    #Calculating scores for the training loop
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)
    
    normal_accuracy = metrics.accuracy_score(correct_labels,predicted_labels)
    average_accuracy = metrics.balanced_accuracy_score(correct_labels,predicted_labels)
    
    TrainLoss.append(trainrunningloss/len(train_dataloader))
    TrainAcc.append(normal_accuracy)
    TrainAvgAcc.append(average_accuracy)
    
    print(f'Train, loss: {trainrunningloss/len(train_dataloader)} Normal acc: {normal_accuracy} Avg acc: {average_accuracy}')
    
    """
    Validation loop, here the validation data is tested on the model
    """
    #Create array and variables needed
    correct_labels = []
    predicted_labels = []
    valrunningloss = 0.0
    
    #Set the network to evaluation mode
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            valloss = criterion(outputs, labels)
            
            valrunningloss += valloss.item()
            
            predictions = outputs.max(dim=1)[1]
        
            correct_labels.append(labels.cpu().numpy())
            predicted_labels.append(predictions.detach().cpu().numpy())
            
            
    ValLoss.append(valrunningloss/len(val_dataloader))
    
    
    #Calculating scores the validation
    predicted_labels = np.concatenate(predicted_labels)
    correct_labels = np.concatenate(correct_labels)
    
    normal_accuracy = metrics.accuracy_score(correct_labels,predicted_labels)
    average_accuracy = metrics.balanced_accuracy_score(correct_labels,predicted_labels)
    
    print(f'Val, loss: {valrunningloss/len(val_dataloader)} Normal acc: {normal_accuracy} Avg acc: {average_accuracy}')
    
    ValAcc.append(normal_accuracy)
    ValAvgAcc.append(average_accuracy)
        
    
PATH = fr'./{ID}_{LEARNINGRATE}_{MOMENTUM}_{BATCH_SIZE}_{EPOCHS}_{NUM_CLASSES}_{NUM_DATASAMPLES}.pth' 
torch.save(net.state_dict(), PATH)
print('Finished Training')
#Saves all the data into a csv file
data = {'Train_Loss': TrainLoss, 'Val_Loss': ValLoss, 'Train_Acc': TrainAcc, 'Train_Avg_Acc': TrainAvgAcc,'Val_Acc': ValAcc,'Val_Avg_Acc': ValAvgAcc}
df = pd.DataFrame(data)

df.to_csv(fr'./{ID}_{LEARNINGRATE}_{MOMENTUM}_{BATCH_SIZE}_{EPOCHS}_{NUM_CLASSES}_{NUM_DATASAMPLES}.csv')
#Trainset prediction
correct_labels = []
predicted_labels = []
data_arr = []
net.eval()
with torch.no_grad():
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        predictions = outputs.max(dim=1)[1]
        
        correct_labels.append(labels.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())


        
predicted_labels = np.concatenate(predicted_labels)
correct_labels = np.concatenate(correct_labels)
decoded_predicted_labels =label_encoder.inverse_transform(predicted_labels)
decoded_correct_labels = label_encoder.inverse_transform(correct_labels)
data = {'predicted labels': predicted_labels, 'correct labels': correct_labels,'decoded predicted labels': decoded_predicted_labels, 'decoded correct labels': decoded_correct_labels}
df = pd.DataFrame(data)
df.to_csv(r'./labels_train.csv')
#Valset prediction
correct_labels = []
predicted_labels = []
net.eval()
with torch.no_grad():
    for i, data in enumerate(val_dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        predictions = outputs.max(dim=1)[1]

        correct_labels.append(labels.cpu().numpy())
        predicted_labels.append(predictions.detach().cpu().numpy())

predicted_labels = np.concatenate(predicted_labels)
correct_labels = np.concatenate(correct_labels)

decoded_predicted_labels =label_encoder.inverse_transform(predicted_labels)
decoded_correct_labels = label_encoder.inverse_transform(correct_labels)
data = {'predicted labels': predicted_labels, 'correct labels': correct_labels,'decoded predicted labels': decoded_predicted_labels, 'decoded correct labels': decoded_correct_labels}

df = pd.DataFrame(data)
df.to_csv(r'./labels_val.csv')
def createGraph(label_csv,images_pr_class,xlabel,ylabel):
    #create class dataframe
    classes = pd.DataFrame(index=label_csv['correct labels'].value_counts().index,columns=('correct','false','images','accuracy'))
    classes['correct'] = 0
    classes['false'] = 0
    classes['images'] = 0
    classes['accuracy'] = 0
    
    for i in range(0,len(label_csv)):
        predicted_class = label_csv['predicted labels'][i]
        correct_class = label_csv['correct labels'][i]
        
        if predicted_class == correct_class:
            classes['correct'].loc[correct_class] +=1
        else:
            classes['false'].loc[correct_class] +=1
            
    for i in enumerate(classes.index,0):
    

        class_correct = classes['correct'].loc[i[1]].item() 
        class_false = classes['false'].loc[i[1]].item() 


        classes['images'].loc[i[1]]  = images_pr_class.loc[i[1]].item()
        classes['accuracy'].loc[i[1]] = (class_correct/(class_correct+class_false))
        
    fig, ax = plt.subplots()
    ax.scatter(classes['accuracy'],classes['images'])
    ax.set_ylabel('Images pr class')
    ax.set_xlabel('Accuracy for train set')
    
    return classes

train_label_csv = pd.read_csv('./labels_train.csv',index_col=0)
val_label_csv = pd.read_csv('./labels_val.csv',index_col=0)

classes_train_stats = createGraph(train_label_csv,groups,'Accuracy for train set','Images pr class')
classes_val_stats = createGraph(val_label_csv ,groups,'Accuracy for validation set','Images pr class')

classes_train_stats
classes_val_stats
#Display image
image_id = train_csv[train_csv['landmark_id']==834].iloc[0].id
image_path = f"/kaggle/input/landmark-recognition-2020/train/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
X = Image.open(image_path)
#Defining the dataset class for how to load the data
class Dataset(torch.utils.data.Dataset):
  def __init__(self, csv_file, image_folder,transform):
        self.csv = csv_file
        self.folder = image_folder
        self.transform = transform

  def __len__(self):
        return len(self.csv)

  def __getitem__(self, index):
        image_id = self.csv.iloc[index].id
        image_path = f"{self.folder}/{image_id[0]}/{image_id[1]}/{image_id[2]}/{image_id}.jpg"
        # Load data and get label
        X = Image.open(image_path)
        y = 0
        X = self.transform(X)
        return X, y
#Test prediction
predictions_arr = []
dataset = Dataset(test_csv,
                        r'/kaggle/input/landmark-recognition-2020/test'
                        ,transform_val
                       )
test_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers=0)
net.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        inputs = data[0].to(device)
        outputs = net(inputs)
        predictions = outputs.max(dim=1)[1]
        
        predictions_arr.append(predictions.detach().cpu().numpy())

predictions_arr = np.concatenate(predictions_arr)
#Make submission file
predictions_arr =label_encoder.inverse_transform(predictions_arr)
data= {'id': test_csv['id'].values, 'landmarks': predictions_arr}
submissionFile = pd.DataFrame(data)
submissionFile.to_csv('./submission.csv')