import torch.nn as nn
import torch
import torch.optim as optim
import pickle
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import TensorDataset, DataLoader,Dataset
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import PIL.Image as Image
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
train_on_gpu = True

model_name = "vgg"
num_classes = 5005
feature_extract = False
print(os.listdir("../input"))
!ls "../input/humpback-whale-identification"
!ls "../input/train-val"
path_train = "../input/train-val/train.csv"
path_val = "../input/train-val/val.csv"

df_train = pd.read_csv("../input/humpback-whale-identification/train.csv")
#df_train = pd.read_csv("../input/train-val/train.csv")
df_val = pd.read_csv("../input/train-val/val.csv")
print(len(df_train))
print(len(df_val))
print(df_train.head())
print(df_val.head())
def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


y_train, le_train = prepare_labels(df_train['Id'])
#train:20286
#val:5075
y_train = y_train[range(20286)]
y_val = y_train[range(5075)]
#y_val, le_val = prepare_labels(df_val['Id'])
print((y_train).shape)
print(y_val.shape)
# For train and val, datafolder = "../input/humpback-whale-identification/train"
# For test, datafolder = "../input/humpback-whale-identification/test"

# For train, datatype = "train"
# For val, datatype = "val"
# For test, datatype = "test"

# For train, y = y_train
# For val, y = y_val

# For train, df = df_train
# For val, df = df_val

# Finished # IMPORTANT: Remember to split y into train and val before creating object of this class
class Whaledataset(Dataset):
    def __init__(self, datafolder, datatype, transform, y, df):
        self.datafolder = datafolder
        self.datatype = datatype
        self.transform = transform
        self.y = y # y is same for both train and val
        if datatype =="train":
            self.df = df.values
        if datatype =="val":
            self.df = df.values
        # self.df not there for test because no df for test        
        self.image_list = [s for s in os.listdir(datafolder)]
        
    def __len__(self):
        if self.datatype == "train":
            return 20286
        elif self.datatype == "val":
            return 5075
        elif self.datatype == "test":
            return len(os.listdir(self.datafolder))
        
        #return len(os.listdir(self.image_list)) #Returns same value for train and val. Correct later
    
    def __getitem__(self, idx):
        if self.datatype =='train' or self.datatype =="val":
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_list[idx])
            label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train' or self.datatype == "val":
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_list[idx]

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)

train_transforms = transforms.Compose([
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomAffine(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
    ])
val_transforms = transforms.Compose([
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(0.5),
                                      transforms.RandomAffine(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
    ])

test_transforms = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                              std = [0.229, 0.224, 0.225])
])

# Write function def accuracy here

def accuracy(outputs,labels):
    preds = outputs
    #print(type(preds), type(outputs))
    #preds_idx = np.argmax(preds, axis =1)
    #labels_idx = np.argmax(labels, axis =1)
    preds_idx = torch.argmax(preds, dim=1)
    ##print(preds_idx)
    labels_idx = torch.argmax(labels, dim=1)
    ##print(labels_idx)
    #preds_values, preds_idx = torch.max(preds, 1)
    #outputs_values, outputs_idx = torch.max(outputs, 1)
    #print(preds.shape)
    #print(labels.shape)
    return (preds_idx==labels_idx).float().sum()

train_datafolder = "../input/humpback-whale-identification/train"
train_datatype = "train"
train_df = df_train
train_transform = train_transforms
train_y = y_train
batch_size = 32
num_workers = 0

val_datafolder = "../input/humpback-whale-identification/train"
val_datatype = "val"
val_df = df_val
val_transform = val_transforms
val_y = y_val
batch_size = 32
num_workers = 0


def train(batch_size = 32, epochs = 10, transform = None):
    tfmd_dataset = Whaledataset(datafolder = train_datafolder, datatype = train_datatype, df = train_df , 
                                transform = train_transform, y = train_y)
    dataloader = torch.utils.data.DataLoader(tfmd_dataset, batch_size = batch_size, shuffle=True, 
                                             num_workers=num_workers)
    
    val_dataset = Whaledataset(datafolder = val_datafolder, datatype = val_datatype, df = val_df , 
                                transform = val_transform, y = val_y)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle=True, 
                                             num_workers=num_workers)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=0.0001)
    
    if torch.cuda.is_available():
        model_ft.cuda()
        nn.DataParallel(model_ft)
        print("GPU")
    epoch_loss_data = []
    epoch_accuracy_data = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    
    for epoch in range(1,epochs+1):  # loop over the dataset multiple times
        i = 0.0
        running_loss, running_loss_total, epoch_accuracy = 0.0, 0.0, 0.0
        val_running_loss, val_running_loss_total, val_epoch_accuracy = 0.0, 0.0, 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.double()
            inputs, labels = inputs.to(device), labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model_ft(inputs).double()
            #print(outputs.size())
            #print(labels.size())
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_accuracy += accuracy(outputs,labels)
            #print(epoch_accuracy)
            #running_epoch_acc = float(epoch_accuracy)/float((labels.size()[0] * (i+1)))
            running_loss += loss.item()
            #running_loss_total += loss.item()
            '''if i%30 == 29:
                print('[%d, %5d] loss: %.3f, epoch_accuracy: %f' % (epoch, i + 1, running_loss, running_epoch_acc))
                running_loss = 0.0'''
            
            '''if i%100 == 99:
                torch.save(model_ft.state_dict(), 'checkpoint')
                torch.save(optimizer.state_dict(), 'optimizer_checkpoint')
                with open('Running_loss.p','wb') as f:
                    pickle.dump(running_loss_total,f)
                with open('Running_accuracy.p','wb') as f:
                    pickle.dump(running_epoch_acc,f)
        epoch_accuracy_data.append(running_epoch_acc)
        epoch_loss_data.append(running_loss_total)'''
            
        epoch_accuracy = float(epoch_accuracy)/float(32 * (i+1))
        print('[%d, %5d] loss: %.3f, epoch_accuracy: %f' % (epoch, i + 1, running_loss, epoch_accuracy))
        running_loss = 0.0
        epoch_accuracy = 0.0
        
        for i, data in enumerate(val_dataloader,0):
            inputs, labels = data
            labels = labels.double()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ft(inputs).double()
            loss = criterion(outputs, labels)
            val_epoch_accuracy += accuracy(outputs,labels)
            val_running_epoch_acc = float(val_epoch_accuracy)/float((labels.size()[0] * (i+1)))
            val_running_loss += loss.item()
            '''if i%30 == 29:
                print('[%d, %5d] val loss: %.3f, val_epoch_accuracy: %f' % (epoch, i + 1, val_running_loss, val_running_epoch_acc))
                val_running_loss = 0.0
            '''
        val_epoch_accuracy = float(val_epoch_accuracy)/float(32 * (i+1))    
        print('[%d, %5d] val loss: %.3f, val_epoch_accuracy: %f' % (epoch, i + 1, val_running_loss, val_epoch_accuracy))
        val_running_loss = 0.0
        val_epoch_accuracy = 0.0
    
train(transform=train_transforms,epochs=25,batch_size=32)
test_datafolder = "../input/humpback-whale-identification/test"
test_datatype = "test"
test_transform = test_transforms
batch_size = 32
num_workers = 0
sub = pd.read_csv("../input/humpback-whale-identification/sample_submission.csv")
def test(batch_size, transform, model):
    test_set = Whaledataset(datafolder = test_datafolder, datatype = test_datatype, transform = test_transform, y = y_train, df = df_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=num_workers)
    model.cuda()
    model.eval()
    for (data, target, name) in test_loader:
        data = data.cuda()
        output = model(data)
        print(output)
        output = output.cpu().detach().numpy()
        for i, (e, n) in enumerate(list(zip(output, name))):
            sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le_train.inverse_transform(e.argsort()[-5:][::-1]))
    sub.to_csv('basic_model.csv', index=False)   
test(batch_size, test_transform, model_ft)
