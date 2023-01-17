from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

df_train = pd.read_csv("../input/train.csv")

data_dir = "../input/train"
model_name = "densenet"
num_classes = 5005
batch_size = 32
num_epochs = 15
feature_extract = False
num_workers = 0

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

y, le = prepare_labels(df_train['Id'])
print(y)

class WhaleDataset(Dataset):
    def __init__(self, datafolder, datatype='train', df=None,
                 transform = transforms.Compose([transforms.ToTensor()]), y=None):
        self.datafolder = datafolder
        self.datatype = datatype
        self.y = y
        if self.datatype == 'train':
            self.df = df.values
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def __getitem__(self, idx):
        if self.datatype == 'train':
            img_name = os.path.join(self.datafolder, self.df[idx][0])
            label = self.y[idx]
            
        elif self.datatype == 'test':
            img_name = os.path.join(self.datafolder, self.image_files_list[idx])
            label = np.zeros((5005,))

        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        if self.datatype == 'train':
            return image, label
        elif self.datatype == 'test':
            # so that the images will be in a correct order
            return image, label, self.image_files_list[idx]



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



data_transforms = transforms.Compose([
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
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])
    ])


def accuracy(outputs,labels):
    preds = outputs.max(dim=1)[1]
    return (preds==labels).float().sum()

def train(batch_size = 32, epochs = 10, transform = None):
    tfmd_dataset = WhaleDataset(datafolder = "../input/train", datatype='train', df = df_train, 
                                transform = transform, y=y)
#     train_sampler = SubsetRandomSampler(list(range(len(os.listdir('../input/train')))))
    dataloader = torch.utils.data.DataLoader(tfmd_dataset, batch_size=batch_size, shuffle=True, 
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

        running_loss, running_loss_total,epoch_accuracy = 0.0, 0.0,0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            labels = labels.double()
            inputs, labels = inputs.to(device), labels.to(device)
        
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model_ft(inputs).double()
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            # print statistics
#             epoch_accuracy += accuracy(outputs,labels)
#             running_epoch_acc = epoch_accuracy/(labels.size()[0] * (i+1))
            running_loss += loss.item()
            running_loss_total += loss.item()
            if i%30 == 29:
                print('[%d, %5d] loss: %.3f, epoch_accuracy:' % (epoch, i + 1,
                                                                    running_loss))
                running_loss = 0.0
            if i%100 == 99:
                torch.save(model_ft.state_dict(), 'checkpoint')
                torch.save(optimizer.state_dict(), 'optimizer_checkpoint')
                with open('Running_loss.p','wb') as f:
                    pickle.dump(running_loss_total,f)
#                 with open('Running_accuracy.p','wb') as f:
#                     pickle.dump(running_epoch_acc,f)
#         epoch_accuracy_data.append(running_epoch_acc)
        epoch_loss_data.append(running_loss_total)
    
train(transform=data_transforms,epochs=28,batch_size=32)
#train(transform=data_transforms,epochs=2,batch_size=32)
sub = pd.read_csv("../input/sample_submission.csv")
def test(batch_size, transform, model):
    test_set = WhaleDataset(datafolder = "../input/test", datatype = "test", transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=num_workers)
    model.eval()
    for (data, target, name) in test_loader:
        data = data.cuda()
        output = model(data)
        print(output)
        output = output.cpu().detach().numpy()
        for i, (e, n) in enumerate(list(zip(output, name))):
            sub.loc[sub['Image'] == n, 'Id'] = ' '.join(le.inverse_transform(e.argsort()[-5:][::-1]))
    sub.to_csv('basic_model.csv', index=False)    

test(32, test_transforms, model_ft)
print(sub.head())
