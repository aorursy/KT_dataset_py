# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import time

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from torch.utils.data import Dataset, DataLoader

import torch

import torch.nn as nn

import torch.optim as optim

from torchvision import transforms, datasets, models



from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training_folder = '/kaggle/input/plant-seedlings-classification/train'
def return_classes(parent_folder):

    classes = {}

    for i,plant_type in enumerate(os.listdir(parent_folder)):

        classes.setdefault(i,plant_type)

    return classes



classes = return_classes(training_folder)

classes
def create_train_dataframe(parent_folder, classes, verbose=True):

    data = []

    for i,plant_class in classes.items():

        folder = os.path.join(parent_folder,plant_class)

        images = os.listdir(folder)

        for image in images:

            image_path = os.path.join(folder,image)

            data.append([image_path,plant_class,i])

            

    df = pd.DataFrame(data,columns=['image','type','class'], index=np.arange(1,len(data)+1))

    return df



training_folder = '/kaggle/input/plant-seedlings-classification/train'

plant_df = create_train_dataframe(training_folder,classes)

print(len(plant_df))

plant_df.head()
def images_per_class(dataframe):

    img_per_class = []

    for i,plant_type in classes.items():

        total_images = len(dataframe[dataframe['class'] == i])

        img_per_class.append(total_images)

        print(plant_type, total_images)

    return img_per_class



images_per_class(plant_df)
def plot_img_per_class(dataframe):

    plt.figure(figsize=(15,10))

    for i in range(12):

        index = np.random.choice

        images = plant_df[plant_df['class']==i]['image'].values

        index = np.random.choice(len(images))

        image = Image.open(images[index])

        plt.subplot(4,3,i+1)

        plt.imshow(image)

        plt.title(classes[i])

        plt.xticks([])

        plt.yticks([])

        

        

plot_img_per_class(plant_df)
## test train split

train_data = plant_df.sample(frac=0.8)

valid_data = plant_df[~plant_df['image'].isin(train_data['image'])]



print(train_data.shape, valid_data.shape)
class Plant_Dataset(Dataset):

    

    def __init__(self, dataframe, transform=None):

        self.dataframe = dataframe

        self.transform = transform



    def __len__(self):

        return len(self.dataframe)

    

    def __getitem__(self,index):

        image_file = self.dataframe.iloc[index,0]

        image = Image.open(image_file).convert('RGB')

        if self.transform is not None:

            image = self.transform(image)

        label = self.dataframe.iloc[index,2]

        

        return image, label
plant_dataset = Plant_Dataset(plant_df)

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # check the theory based on the normalizing the image

])



validation_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # check the theory based on the normalizing the image

])



image, label = plant_dataset.__getitem__(1)

train_transform(image)
## make dataset and dataloaders for train and test set respectively

datasets =  {}



datasets['train'] = Plant_Dataset(train_data, train_transform)

datasets['validation'] = Plant_Dataset(valid_data, validation_transform)



batch_size=32



dataloaders = {}

dataloaders['train'] = DataLoader(datasets['train'], shuffle=True, batch_size=batch_size)

dataloaders['validation'] = DataLoader(datasets['validation'], shuffle=False, batch_size=batch_size)



dataset_sizes = {}

dataset_sizes['train'] = len(datasets)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
model = models.resnet50(pretrained=True)

#I recommend training with these layers unfrozen for a couple of epochs after the initial frozen training

for param in model.parameters():  ## freezing the initial layers

    param.requires_grad = False

    

model.fc = nn.Sequential(

    nn.Linear(model.fc.in_features,12)

)



loss_fn = nn.CrossEntropyLoss()

opt = optim.Adam(model.parameters())

# lr scheduler - reduce on loss plateau decay

# lr = lr * factor 

# mode (str) – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

# in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min

# patience: number of epochs - 1 where loss plateaus before decreasing LR

        # patience = 0, after 1 bad epoch, reduce LR

# factor = decaying factor

scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.1, verbose=True, mode='max')
images,labels = next(iter(dataloaders['train']))

images,labels = images.to(device),labels.to(device)

model = model.to(device)

torch.argmax(model(images),dim=1)
def train(model, optimizer, loss_fn, epochs=10, device=device):

    start_time = time.time()

    best_acc = 0

    best_model_wts = model.state_dict()

    

    train_loss = []

    val_loss = []

    

    train_acc = []

    val_acc = []

    

    model = model.to(device)

    for epoch in tqdm_notebook(range(epochs)):

        print('Epoch {}/{}'.format(epoch,epochs))

        print('-'*20)

        

        for phase in ['train', 'validation']:

            

            ## 1. setting up the training mode 

            if phase == 'train':

                model.train(True)

            else:

                model.train(False)

                

            running_loss = 0.0

            running_corrects = 0.0

            running_batch = 0.0

            

            for data in dataloaders[phase]:

                images,labels = data

                images,labels = images.to(device), labels.to(device)

                

                ## zero the parameter gradients

                optimizer.zero_grad()



                # forward pass

                output = model(images)

                loss = loss_fn(output,labels)

                preds = torch.argmax(output,dim=1) ## for calculating the running corrects



                if phase == 'train':

                    loss.backward()

                    optimizer.step()



                # statistics

                running_loss += loss.item()

                running_corrects += torch.sum(preds==labels)

                running_batch += 1

            

            epoch_loss = running_loss/running_batch

            epoch_accuracy = running_corrects/len(datasets[phase])

            

            # store the statistics for plotting and step lr scheduler

            if phase == 'train':

                train_loss.append(epoch_loss)

                train_acc.append(epoch_accuracy)

            else:

                val_loss.append(epoch_loss)

                val_acc.append(epoch_accuracy)

                scheduler.step(epoch_accuracy) ## step the learning rate in validation phase to avoid overfitting of the dataset

                

            print('{} Loss {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            

            ## save the best model

            if phase == 'validation' and epoch_accuracy > best_acc:

                best_acc = epoch_accuracy

                best_model_wts = model.state_dict()

                

        

                

    time_elapsed = time.time() - start_time    

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed % 60))

    print('Best accuracy {:.4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)

    

    

    ## plot the statistics

    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, sharex=True)

    ax1.plot(train_loss)

    ax1.plot(val_loss)

    ax1.set_title('Cross Entropy loss')

    ax1.set_xlabel('Epochs')

    

    ax2.plot(train_acc)

    ax2.plot(val_acc)

    ax2.set_title('Accuracy (%)')

    ax2.set_xlabel('Epochs')

    

    return model
%%time

trained_model = train(model,opt, loss_fn, device=device, epochs=100)
## save the model

state_dict = trained_model.state_dict() 

torch.save(state_dict, 'model.pth')
images_per_class(valid_data)
def plot_confusion_matrix(model, dataloader, device='cpu'):

    y_true = []

    y_preds = []

    model = model.to(device)

    for images,labels in tqdm_notebook(dataloader):

        images = images.to(device)

        y_true.extend(labels.numpy())

        with torch.no_grad():

            output = model(images)

            pred = torch.argmax(output,dim=1)

            y_preds.extend(pred.cpu().numpy())



    cm = confusion_matrix(y_true,y_preds)

    df_cm = pd.DataFrame(cm, index=list(classes.values()), columns = list(classes.values()))

    plt.figure(figsize=(10,5))

    ax = sns.heatmap(df_cm, annot=True, fmt='d')

    ax.set_xlabel('Predicted')

    ax.set_ylabel('True Labels')

    plt.show()

    

    return cm



def evaluation_statistics(cm):

    for j,plant in tqdm_notebook(classes.items()):

        pred_positive_labels = np.sum(cm[:,j])

        true_labels = np.sum(cm[j,:])

        true_positives = cm[j,j]



        precision = true_positives/pred_positive_labels

        recall = true_positives/true_labels

        f_score = 2*precision*recall/(precision+recall)

        accuracy = true_positives/(true_labels+pred_positive_labels-true_positives)



        print('{} | TP = {} | Predicted Yes = {} | True Labels = {}'.format(plant,true_positives, pred_positive_labels, true_labels))

        print('Precision {:.2f}'.format(precision)) # Out of all predicted positive how many are actually correct

        print('Recall {:.2f}'.format(recall)) # True Positive Rate

        print('F_score {:.2f}'.format(f_score))

        print('Accuracy {:.2f}'.format(accuracy)) # True Positive Rate

        print('-'*20)
%%time

# evaluate_data

datasets['train'] = Plant_Dataset(train_data, validation_transform)

train_dataloader= DataLoader(datasets['train'], shuffle=False, batch_size=32)



train_cm = plot_confusion_matrix(trained_model,train_dataloader,device=device)

evaluation_statistics(train_cm)
%%time 

datasets['validation'] = Plant_Dataset(valid_data, validation_transform)

val_dataloader= DataLoader(datasets['validation'], shuffle=False, batch_size=32)



val_cm = plot_confusion_matrix(trained_model,val_dataloader,device=device)

evaluation_statistics(val_cm)
%%time 

test_folder = '/kaggle/input/plant-seedlings-classification/test'



def predict_test_images(folder, model, device='cpu'):

    classification = []

    model = model.to(device)

    for image_file_name in tqdm_notebook(os.listdir(folder)):

        image_path = os.path.join(folder,image_file_name)

        image = Image.open(image_path)

        image_input = validation_transform(image).unsqueeze(0).to(device)

        

        ## prediction

        with torch.no_grad():

            output = model(image_input)

            pred = torch.argmax(output,dim=1).item()

            classification.append([image_file_name,classes[pred]])

            

    return classification



test_classification = predict_test_images(test_folder, trained_model, device=device)
submission = pd.DataFrame(np.array(test_classification), columns= ['file','species'], index=np.arange(1,len(test_classification)+1))

submission.head()
submission.to_csv('submission.csv',index = False)