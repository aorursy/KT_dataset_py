# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

from glob import glob

import seaborn as sns

from torch.utils import data

import torch

from PIL import Image

torch.manual_seed(42)

np.random.seed(42)
#Assigining te path of the dataset directory

base_skin_dir = os.path.join('..', 'input')

#Craeting a dictionary of keys image-id's  and with values of each resective .jpg image path

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x

                     for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}
#Creating dictory of more firenly and user convininet disease names with respective short for names in the data frame

lesion_type_dict = {

    'nv': 'Melanocytic nevi',

    'mel': 'dermatofibroma',

    'bkl': 'Benign keratosis-like lesions ',

    'bcc': 'Basal cell carcinoma',

    'akiec': 'Actinic keratoses',

    'vasc': 'Vascular lesions',

    'df': 'Dermatofibroma'

}
#Reading the csv file using pandas by pd.read_csv(directory name, csv file name) function giving theath of the csv file.

tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))

#Adding a column called 'path' getting the values of corresponding path of the respective image_id present in the table and path from 'imageid_path_dict' dictionary which we created before..

tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)

#Adding a column called 'cell_type' which contains the respective names of the diseases which are corresponding to the 'dx' column in the dataframe

tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get) 

tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
tile_df[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()
#Knowing what the colun 'cell_type' is for and how much each varibles occurs in the whole dataset.

tile_df['cell_type'].value_counts()
#Having a quick glance of first rows in the data frame.

tile_df.head()
# PyTorch has a nice feature that it offers well established models. 

#These models can optionally allready be pretrained on the ImageNet dataset,causing the training time to be lower in general.

#Let us load a pretrained ResNet50 and adjust the last layer a little bit.

import torchvision.models as models

model_conv = models.resnet50(pretrained=True)
#printign the model to visulasie the layers present

print(model_conv)
#THis is to know about the model Input and Output featurs is there and bias.

#We are printing this to know beacause its a pretrained model.

#We need to known on what inputs it is trained and how many outputs features are there now and so on..

print(model_conv.fc)
# Define the device:

device = torch.device('cuda:0')



# Put the model on the device:

model = model_conv.to(device)
#Importing Sklearn.model_selector which has a nice function called 'train_test_split()' which would help is splitting the current data into train and valiadtion data samples.

from sklearn.model_selection import train_test_split

#Here we split the  whole data into train-90% and validation-10% 

#This indicates that the datset is divided traing part as train_df(90%) and validation(10%) as test_df

train_df, test_df = train_test_split(tile_df, test_size=0.1)
# We can split the test set again in a validation set and a true test set:

validation_df, test_df = train_test_split(test_df, test_size=0.5)
#When we reset the indexof namy row, the old index is added as a column, and a new sequential index is used.

#Using reset_index() for train data

train_df = train_df.reset_index()

#Using reset_index() for validation data

validation_df = validation_df.reset_index()

#Using reset_index() for test data

test_df = test_df.reset_index()
class Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'

    def __init__(self, df, transform=None):

        'Initialization'

        self.df = df

        self.transform = transform



    def __len__(self):

        'Denotes the total number of samples'

        return len(self.df)



    def __getitem__(self, index):

        'Generates one sample of data'

        # Load data and get label

        X = Image.open(self.df['path'][index])

        y = torch.tensor(int(self.df['cell_type_idx'][index]))



        if self.transform:

            X = self.transform(X)



        return X, y
# Define the parameters for the dataloader.

#batch_size is number of images per one iteration and shuffle is shuffle all the images in each iteration.

#num_woorkers is nuber of sub processes you wanna run parallely.

#Here 'batch_size': 4,'shuffle': True,'num_workers': 6 keepinf these all in a dictionary called params.

params = {'batch_size': 4,

          'shuffle': True,

          'num_workers': 6}
# define the transformation of the images.

import torchvision.transforms as trf

#we only perform mirroring (RandomHorizontalFlip, RandomVerticalFlip), Crop the image to the image center, where the melanom is most often located (CenterCrop), 

#randomly crop from the center of the image (RandomCrop) and normalize the image according to what the pretrained model needs (Normalize). 

#We then transform the image to a tensor using, which is required to use it for learning with PyTorch, with the function ToTensor:

composed = trf.Compose([trf.RandomHorizontalFlip(), trf.RandomVerticalFlip(), trf.CenterCrop(256), trf.RandomCrop(224),  trf.ToTensor(),

                        trf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# Define the trainingsset using the table train_df and using our defined transitions (composed)

training_set = Dataset(train_df, transform=composed)

training_generator = data.DataLoader(training_set, **params)



# Same for the validation set:

validation_set = Dataset(validation_df, transform=composed)

validation_generator = data.DataLoader(validation_set, **params)
#In our case this will be an Adam optimizer with a learning rate of  1e−6/2.6. 

optimizer = torch.optim.Adam(model_conv.parameters(), lr=1e-6/2.6)

#The criterion or the loss function that we will use is the CrossEntropyLoss.

criterion = torch.nn.CrossEntropyLoss()

#This is a typical chosen loss function for multiclass classification problems.

#Taking 500 epochs for traning

max_epochs = 500

#Declaring a empty list called training_error

trainings_error = []

#Declaring a empty list called validation_error

validation_error = []

#Iterating in range of epochs

for epoch in range(max_epochs):

    #Printing epochs

    print('epoch:', epoch)

    #Taking initially count_train as zero

    count_train = 0

    #Taking an empty list called trainings_error_tmp

    trainings_error_tmp = []

    #Calling train function of class model

    model.train()

    #Iterating through validation_geneartor as two parts called data_sample, y

    for data_sample, y in training_generator:

        #Put the model onto device

        data_gpu = data_sample.to(device)

        #Put the model onto device

        y_gpu = y.to(device)

        #Traing the model on the data_gpu

        output = model(data_gpu)

        #Calculating the Loss function

        err = criterion(output, y_gpu)

        #Back propoagting according to the error generated

        err.backward()

        #Using optimisation to optimize the problem

        optimizer.step()

        #Appending the err.item() value to trainings_error_tmp list

        trainings_error_tmp.append(err.item())

        #Increament count_train variable by one

        count_train += 1

        #Check the condition wheather count>=100

        if count_train >= 100:

            #Initialise a variable count_train to zero 

            count_train = 0

            #Takig mean of all the values in the list trainings_error_tmp and assign that to mean_training_error

            mean_trainings_error = np.mean(trainings_error_tmp)

            #Append mean_trainings_error to list mean_trainings_error)

            trainings_error.append(mean_trainings_error)

            #Printing the training eroor.

            print('trainings error:', mean_trainings_error)

            #Breking the loop after that

            break

    #Setting Gradinent to False.

    with torch.set_grad_enabled(False):

        #Declaring an empty list called validation_error_tmp 

        validation_error_tmp = []

        #Initailse coount_val to zero

        count_val = 0

        #Calling eval() function that is making model ready for evalution.

        model.eval()

        #Iterating through validation_geneartor as two parts called data_sample, y

        for data_sample, y in validation_generator:

            data_gpu = data_sample.to(device)

            y_gpu = y.to(device)

            output = model(data_gpu)

            #Calculating the los function using the crossEntropyLoss Function.

            err = criterion(output, y_gpu)

            #Appending err.item() value to the list validation_error_tmp

            validation_error_tmp.append(err.item())

            #Increamenting count_val by one

            count_val += 1

            #Checking Condition wheather count_val >= 10

            if count_val >= 10:

                #Initialising a variable called count_val

                count_val = 0

                #Taking the mean of all the values of validation_error_tmp

                mean_val_error = np.mean(validation_error_tmp)

                #Appending the value of mean_val_error to list validation_error

                validation_error.append(mean_val_error)

                #Printing the validation in each iteration

                print('validation error:', mean_val_error)

                break
#Plotting the traning_error on a graph

plt.plot(trainings_error, label = 'training error')

#Plotting the validation_error on a graph

plt.plot(validation_error, label = 'validation error')

plt.legend()

#Showing plot

plt.show()

#Making the model ready for evaluation

model.eval()

#Checking on the model validation dataset

test_set = Dataset(validation_df, transform=composed)

test_generator = data.SequentialSampler(validation_set)
#Declaring and emmpty list called results_array

result_array = []

#Declaring and emmpty list called results_array

gt_array = []

#Iterate through the test_genartor

for i in test_generator:

    #validation_set.__getitem__(i) return two values storing them in data_sample, y respectively. 

    data_sample, y = validation_set.__getitem__(i)

    data_gpu = data_sample.unsqueeze(0).to(device)

    #testing the model on data_gpu and store the return value in output

    output = model(data_gpu)

    #Usin torch.argmax finding the value and string it in the result

    result = torch.argmax(output)

    #Appending the result.item() value to result_array

    result_array.append(result.item())

    #Appending the y.item() value to gt_array

    gt_array.append(y.item())
#Taking 1 if its correct prediction correct prediction els ezero if its wrong prediction in correct_results array

correct_results = np.array(result_array)==np.array(gt_array)
#Summing up all the values in orrect_reults to know number of correct predictions 

sum_correct = np.sum(correct_results)
#Calculating Accuracy as ratio of number of correct prediction to total number of predictions

accuracy = sum_correct/test_generator.__len__()
#Printing the accuracy of the model

print(accuracy)