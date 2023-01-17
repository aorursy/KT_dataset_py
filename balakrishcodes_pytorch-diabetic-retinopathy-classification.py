# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import torch 

import matplotlib.pylab as plt

import numpy as np

import pandas as pd

import time, os, random

import h5py

from torch.utils.data import Dataset, DataLoader

from keras.utils import to_categorical

from torchvision import transforms

print(torch.__version__)

import nibabel as nib

from torch.autograd import Variable

import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torchvision import datasets, transforms, models

from torch.utils.data import Dataset, DataLoader

import torchvision

from torchvision import transforms, utils

#!pip install torchsummary --quiet

!pip install torchsummaryX  --quiet

from torchsummaryX import summary
def func(df):

    return os.path.join('/kaggle/input/retinopathy-train-2015/rescaled_train_896/', df.image+".png")

    

df = pd.read_csv('/kaggle/input/retinopathy-train-2015/trainLabels.csv')

df['image_path'] = df.apply(func, axis=1)

#df.to_csv('/kaggle/working/DR.csv',index=False)

df.head()
for i in range(5):

    print("label {} - Total Count {}".format(i,df.level[df.level==i].count()))
import seaborn as sns

sns.countplot(df['level'])
df_final = pd.DataFrame()

sample = 5500 # Provide your choice of number of samples per class



for i in range(5):

    min_val = len(df[df.level==i])

    temp_df = df[df.level==i].sample(min(sample,min_val))

    df_final = df_final.append(temp_df, ignore_index = True)

    print("Extracted {} samples from label/level {}".format(len(temp_df), i))

    

print()

print(df_final.shape)

df_final.head()

fold = ['train']*(int(len(df_final)*0.9)) + ['valid']*(len(df_final) - int(len(df_final)*0.9))

random.shuffle(fold)

df_final['fold'] = fold

df_final.head()
import seaborn as sns

sns.countplot(df_final['level'])
for i in range(5):

    train = valid = 0

    train = df_final[(df_final['level'] == i) & (df_final['fold'] =="train")].shape[0]

    valid = df_final[(df_final['level'] == i) & (df_final['fold'] =="valid")].shape[0]

    print("For level {}, total number of training samples is {} and testing samples is {}".format(i, train, valid))

    print()
df_final.to_csv('/kaggle/working/DR.csv',index=False)
NUM_SAMP=5

fig = plt.figure(figsize=(25, 16))

import cv2

IMG_SIZE = 512

for jj in range(5):

    for i, (idx, row) in enumerate(df_final.sample(NUM_SAMP,random_state=123+jj).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, jj * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/retinopathy-train-2015/rescaled_train_896/{row['image']}.png"

        image = plt.imread(path)

        plt.imshow(image)

        ax.set_title('%d-%s' % (idx, row['image']) )
def load_ben_color(path, IMG_SIZE, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)

    

    image = image.astype(np.float32) #

    image /= 255. #

    return image, IMG_SIZE



def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img
path1 = "../input/retinopathy-train-2015/rescaled_train_896/10003_left.png"

image = cv2.imread(path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape, type(image), image.dtype)

image = crop_image_from_gray(image)

print(image.shape, type(image), image.dtype)

image = cv2.resize(image, (512, 512))

image = cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX=10) ,-4 ,128)

image = image.astype(np.float32)

image /= 255.

print(image.shape, type(image), image.dtype)

plt.imshow(image)

plt.show()
NUM_SAMP=5

fig = plt.figure(figsize=(25, 16))

import cv2

IMG_SIZE = 512

for jj in range(5):

    for i, (idx, row) in enumerate(df_final.sample(NUM_SAMP,random_state=123+jj).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, jj * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/retinopathy-train-2015/rescaled_train_896/{row['image']}.png"

        image, _ = load_ben_color(path,IMG_SIZE,sigmaX=30)  

        #image = np.array(image, dtype="float32")

        plt.imshow(image)

        ax.set_title('%d-%s' % (idx, row['image']) )
classifier = True # input as False makes the model regressor.



# Flag for feature extracting. When False, we finetune the whole model,

#   when True we only update the reshaped layer params

feature_extract = False
# Number of classes in the dataset

# https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/

if classifier:

    num_classes = 5 # Classifier

    criterion =  nn.CrossEntropyLoss() 

else:

    num_classes = 1 # Regressor

    criterion =  nn.MSELoss() 







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

        """ Resnet18

        """

        model_ft = models.resnet18(pretrained=use_pretrained)

        set_parameter_requires_grad(model_ft, feature_extract)

        num_ftrs = model_ft.fc.in_features

        model_ft.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

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

model_name = "resnet" # Models to choose ["resnet", "alexnet", "vgg", "squeezenet", "densenet", "inception"]

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)



# Print the model we just instantiated

print(model_ft)

print()

print("Input image size format",(input_size,input_size))
summary(model_ft, torch.zeros((1, 3, input_size, input_size)))
# Flag for feature extracting. When False, we finetune the whole model,

#   when True we only update the reshaped layer params

feature_extract = True



BATCH_SIZE =  32 # Desired batch size

SAMPLE = 0 # Increase the sample size if you want to train only on a specific number of samples, otherwise to train on entire datset, set sample = 0

img_size = input_size # This sets the input image size based on the model's you choose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



print("Running on",device)

model_ft = model_ft.to(device)



params_to_update = model_ft.parameters()

print("Params to learn:")

if feature_extract:

    params_to_update = []

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            params_to_update.append(param)

            print("\t",name)

else:

    for name,param in model_ft.named_parameters():

        if param.requires_grad == True:

            print("\t",name)





learning_rate=0.01

# optimizer = optim.Adam(params_to_update, lr=learning_rate)

optimizer = optim.SGD(params_to_update, lr=learning_rate , momentum=0.9)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.85, patience=2, verbose=True)
class DRDataset(Dataset):

    # Constructor

    def __init__(self, csv_file, csv_dir, fold , img_size, transform=None, sample=0):

        # Image directory

        self.transform = transform

        self.img_size = img_size

        self.fold = fold

        self.df = pd.read_csv(os.path.join(csv_dir , csv_file), index_col=0)

        self.df = self.df[self.df['fold'] == fold]

        self.sample = sample

        if self.sample > 0:

            self.df = self.df.sample(self.sample, random_state=42) # sample



    # Get the length

    def __len__(self):

        return len(self.df)

    

    # Getter

    def __getitem__(self, idx):

        path = self.df.image_path[idx]

        image, size = load_ben_color(path,sigmaX=30, IMG_SIZE=self.img_size)

        assert size == self.img_size

        image = np.array(image)

        image = torch.from_numpy(image)

        

        label = np.array(df.level[idx])

        label = label.astype(np.uint8)

        label = torch.from_numpy(label)

        return image, label
transformed_datasets = {}

transformed_datasets['train'] = DRDataset(csv_file = "DR.csv",csv_dir = "/kaggle/working",  fold="train" ,img_size = img_size, sample=SAMPLE)

transformed_datasets['valid'] = DRDataset(csv_file = "DR.csv", csv_dir= "/kaggle/working",  fold="valid" ,img_size = img_size, sample=SAMPLE)

 

dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(transformed_datasets['train'],batch_size=BATCH_SIZE,shuffle=True,num_workers=8)

dataloaders['valid'] = torch.utils.data.DataLoader(transformed_datasets['valid'],batch_size=BATCH_SIZE,shuffle=True,num_workers=8)  

print()

print(len(dataloaders['train']))

print(len(dataloaders['valid']))
for data in dataloaders['valid']:

    images, labels = data

    images = images.to('cpu')

    print(labels, labels.shape)

    break

plt.figure(figsize=(20,10)) 

for i in range(16):

    plt.subplot(4,4, i+1)

    plt.imshow(images[i,:,:,:])
from IPython.display import HTML, display

 

class ProgressMonitor(object):

    """

    Custom IPython progress bar for training

    """

    

    tmpl = """

        <p>Loss: {loss:0.4f}   {value} / {length}</p>

        <progress value='{value}' max='{length}', style='width: 100%'>{value}</progress>

    """

 

    def __init__(self, length):

        self.length = length

        self.count = 0

        self.display = display(self.html(0, 0), display_id=True)

        

    def html(self, count, loss):

        return HTML(self.tmpl.format(length=self.length, value=count, loss=loss))

        

    def update(self, count, loss):

        self.count += count

        self.display.update(self.html(self.count, loss))



def checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_valid_loss):

    print('saving')

    print()

    state = {'model': model,'best_loss': best_loss,'epoch': epoch,'rng_state': torch.get_rng_state(), 'optimizer': optimizer.state_dict(),}

    torch.save(state, '/kaggle/working/checkpoint-DR')

    torch.save(model.state_dict(),'/kaggle/working/checkpoint-statedict-DR')

    

def train_new(model,criterion,optimizer,num_epochs,dataloaders,dataset_sizes,first_epoch=1):

    since = time.time() 

    best_loss = 999999

    best_epoch = -1

    last_train_loss = -1

    plot_train_loss = []

    plot_valid_loss = []

    plot_train_acc = []

    plot_valid_acc = []

 

 

    for epoch in range(first_epoch, first_epoch + num_epochs):

        print()

        print('Epoch', epoch)

        running_loss = 0.0

        valid_loss = 0.0

        training_accuracy = 0

        validation_accuracy = 0

      

        # train phase

        model.train(True)

 

      # create a progress bar

        progress = ProgressMonitor(length=dataset_sizes["train"])

 

        for data in dataloaders["train"]:

            inputs, labels  = data # (Batch_size, width, height, channels)

            batch_size = inputs.shape[0]

            inputs = inputs.permute(0,3,1,2) # Batch_size, channels, width, height

            inputs = inputs.to(device)

            if classifier:

                labels = labels.to(device,dtype=torch.long)

            else:

                labels = labels.to(device,dtype=torch.float).view(-1, 1)

            inputs = Variable(inputs)

            labels = Variable(labels)

 

            # clear previous gradient computation

            optimizer.zero_grad()

            outputs = model(inputs) # batch, 2, 240, 240



            loss = criterion(outputs, labels)

 

            loss.backward()

            optimizer.step()

                      

            running_loss += loss.data * batch_size

            if classifier:

                training_accuracy += (outputs.argmax(1) == labels).sum().item()

            else:

                training_accuracy += (outputs.round().int() == labels).sum().item()

          # update progress bar

            progress.update(batch_size, running_loss)

 

        epoch_loss = running_loss / dataset_sizes["train"]

        

        print('Training Accuracy is {} and Training loss {}'.format(training_accuracy / dataset_sizes["train"],epoch_loss.item()))

        plot_train_loss.append(epoch_loss)

        plot_train_acc.append(training_accuracy / dataset_sizes["train"])

 

 

      # validation phase

        model.eval()

      # We don't need gradients for validation, so wrap in 

      # no_grad to save memory

        with torch.no_grad():

            for data in dataloaders["valid"]:

                inputs, labels  = data

                batch_size = inputs.shape[0]

                inputs = inputs.permute(0,3,1,2)

                inputs = inputs.to(device)

                if classifier:

                    labels = labels.to(device,dtype=torch.long)

                else:

                    labels = labels.to(device,dtype=torch.float).view(-1, 1)

                inputs = Variable(inputs)

                labels = Variable(labels)

                

                outputs = model(inputs)

 

            # calculate the loss

                optimizer.zero_grad()

                loss = criterion(outputs, labels)

            

            # update running loss value

                valid_loss += loss.data * batch_size

                if classifier:

                    validation_accuracy += (outputs.argmax(1) == labels).sum().item()

                else:

                    validation_accuracy += (outputs.round().int() == labels).sum().item()

                    

 

        epoch_valid_loss = valid_loss / dataset_sizes["valid"]

        scheduler.step(epoch_valid_loss)

        print('Validation Accuracy is {} and Validation loss {}'.format(validation_accuracy / dataset_sizes["valid"],epoch_valid_loss.item()))

        plot_valid_loss.append(epoch_valid_loss)

        plot_valid_acc.append(validation_accuracy / dataset_sizes["valid"])

          

        if epoch_valid_loss < best_loss:

            best_loss = epoch_valid_loss

            best_epoch = epoch

            checkpoint_and_save(model, best_loss, epoch, optimizer, epoch_valid_loss.item()) 

        if ((epoch - best_epoch) >= 10):

            print("no improvement in 10 epochs, break")

            break

 

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

 

    return plot_train_loss, plot_valid_loss, plot_train_acc, plot_valid_acc, model
%%time

dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'valid']}

epochs = 16

if __name__=="__main__":

    train_losses, valid_losses, train_accuracy, valid_accuracy, model = train_new(model = model_ft ,criterion = criterion,optimizer = optimizer,

                                                                                  num_epochs=epochs,dataloaders = dataloaders,

                                                                                  dataset_sizes = dataset_sizes)
# Plot Accuracy

plt.title('Training and Validation Accuracy')

plt.plot(train_accuracy)

plt.plot(valid_accuracy)

plt.legend(['Training_Accuracy','Validation_Accuracy'])
plt.title('Training and Validation Loss')

plt.plot(train_losses)

plt.plot(valid_losses)

plt.legend(['Training_loss','Validation_loss'])
plt.figure(figsize=(12,8))

plt.title('Training & Validation Accuracy and Loss')

plt.plot(train_accuracy)

plt.plot(valid_accuracy)

plt.plot(train_losses)

plt.plot(valid_losses)

plt.legend(['Training_Accuracy','Validation_Accuracy','Training_loss','Validation_loss'])
def tester(image, model):

    img = torch.from_numpy(image).to(device)

    img = img.unsqueeze(0)

    img = img.permute(0,3,1,2) # (bs, width, height, channels) --> (bs, channels, width, height)

    output = model(img)

    if classifier:

        return output.argmax(1).item()

    else:

        return output.round().item()

    

NUM_SAMP=10

fig = plt.figure(figsize=(25, 16))

count = 0

for jj in range(5):

    for i, (idx, row) in enumerate(df_final[df_final.fold=="valid"].sample(NUM_SAMP,random_state=123+jj).iterrows()):

        ax = fig.add_subplot(5, NUM_SAMP, jj * NUM_SAMP + i + 1, xticks=[], yticks=[])

        path=f"../input/retinopathy-train-2015/rescaled_train_896/{row['image']}.png"

        orig_label = int(row['level'])

        image, _ = load_ben_color(path,img_size,sigmaX=30) #224,224,3

        pred_label = tester(image, model)

        if (orig_label == pred_label):

            count +=1

        plt.imshow(image)

        ax.set_title('%d - %d' % (orig_label, pred_label))

print()

print("Out of {} samples, model predicted {} samples correctly".format((NUM_SAMP*5), count))

print()