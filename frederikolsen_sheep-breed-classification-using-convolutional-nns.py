# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# also load the following for this project
from collections import Counter
import random  # import random library
import PIL  # import Pillow library
from PIL import Image  # import Image functionality
import torch  # import pytorch
from torch import nn
from sklearn import metrics  # import evaluation metrics


# import seaborn and matplotlib.pyplot for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
image_dir = []
import os
for dirname, _, filenames in os.walk('/kaggle/input/sheep-breed-classification/SheepFaceImages/'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        image_dir.append(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
images = []  # variable to store the image data as 3D numpy arrays
labels = []  # variable to store the label of each image
unique_labels = list(set([str(directory[57:].split('/')[0]) for directory in image_dir]))  # unique labels
for d in image_dir:  # iterate over file directory
    if np.asarray(Image.open(d)).shape == (181, 156, 3):  # serves to remove different sized images
        images.append(np.asarray(Image.open(d)) / 255)  # append images as scaled 3D numpy arrays
        labels.append(unique_labels.index(d[57:].split('/')[0]))  # append classifications as encoded labels
# a function that will shuffle the data
def shuffle_data(x_vals, y_vals):

    # convert the x and y values to tuples, and then shuffle it to randomise the order of variables
    data = [(x, y) for x, y in zip(x_vals, y_vals)]
    random.shuffle(data)
    new_x_vals, new_y_vals = zip(*data)
    
    return list(new_x_vals), list(new_y_vals)

# a function to tensorize the data
def tensorize_data(x_vals, y_vals):
    
    x_tensor = []
    y_tensor = []
    for x, y in zip(x_vals, y_vals):
        x_tensor.append(torch.from_numpy(x).float())
        y_tensor.append(torch.from_numpy(np.array(y)).long())
                
    return x_tensor, y_tensor
# shuffle the data
x_values, y_values = shuffle_data(images, labels)

# tensorize the data
x_tensor, y_tensor = tensorize_data(x_values, y_values)
# split the train/test split
split = 279 # define the index upon which to split
x_train = x_tensor[:-split]
y_train = y_tensor[:-split]
x_test = x_tensor[-split:]
y_test = y_tensor[-split:]
print(len(x_train), len(x_test))
# define our sheep classification model
class SheepClassifier(nn.Module):   
    def __init__(self):
        super(SheepClassifier, self).__init__()

        # the CNN component
        self.cnn_component = nn.Sequential(
            nn.Conv2d(181, 70, kernel_size=2),  #  2D convolution
            nn.BatchNorm2d(70),  # batch normalization
            nn.ReLU(inplace=True),  # ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # 2D max pooling
        )

        # the MLP fully connected layer
        self.cnn_output_linear = nn.Sequential(
            nn.Dropout(0.02),
            nn.Linear(70 * 77 * 1, 800),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(800, 200),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(200, 75),
            nn.ReLU(),
            nn.Linear(75, 4)
        )

    # defining the forward pass    
    def forward(self, x):
        
        # pass through the CNN component
        cnn_output = self.cnn_component(x)
        
        # flatten outputs
        cnn_output_reshaped = cnn_output.view(cnn_output.shape[0], cnn_output.shape[1]
                                              * cnn_output.shape[2] * cnn_output.shape[3])
        
        # pass flattened output through the MLP fully connected layer
        output = self.cnn_output_linear(cnn_output_reshaped)
        
        return output
    
sheep_classifier = SheepClassifier()  # define the model
print(sheep_classifier)  # print the model to summarise it
loss_fn = nn.CrossEntropyLoss()  # loss function
optimizer = torch.optim.Adam(sheep_classifier.parameters(), lr=0.000025)  # optimizer
epochs = 100  # number of times we repeat the training process
batch_size = 50  # batch number

x_val, y_val = x_train[-batch_size:], y_train[-batch_size:]  # validation set 
x_train, y_train = x_train[:-batch_size], y_train[:-batch_size]  # new training set

# form validation batch in advance
val_x_batch = None
val_y_batch = None
for v in range(batch_size):
    if v % batch_size == 0:
        val_x_batch = torch.unsqueeze(x_val[v], dim=0)
        val_y_batch = torch.unsqueeze(y_val[v], dim=0)
    else:
        val_x_batch = torch.cat((val_x_batch, torch.unsqueeze(x_val[v], dim=0)), dim=0)
        val_y_batch = torch.cat((val_y_batch, torch.unsqueeze(y_val[v], dim=0)), dim=0)

# now start the training process
for i in range(epochs):
    x_batch = None
    y_batch = None
    for z in range(len(x_train)):
        # batching done within the training process
        if z % batch_size == 0:
            x_batch = torch.unsqueeze(x_train[z], dim=0)
            y_batch = torch.unsqueeze(y_train[z], dim=0)
        else:
            x_batch = torch.cat((x_batch, torch.unsqueeze(x_train[z], dim=0)), dim=0)
            y_batch = torch.cat((y_batch, torch.unsqueeze(y_train[z], dim=0)), dim=0)            
        
        if x_batch.shape[0] == batch_size:
            preds = sheep_classifier(x_batch)  # feed batch to the model
            single_loss = loss_fn(preds, y_batch)  # calculate loss from the batch
            
            # validation set evaluation
            with torch.no_grad():  # disable autograd engine
                val_preds = sheep_classifier(val_x_batch)  # feed val batch to the model
                val_loss = loss_fn(val_preds, val_y_batch)  # calculate loss from the val batch

            optimizer.zero_grad()  # zero the gradients
            single_loss.backward()  # backpropagate through the model
            optimizer.step()  # update parameters
        
    if i%5 == 0:
        print(f'epoch: {i:5} training loss: {single_loss.item():10.8f} validation loss: {val_loss.item():10.8f}')
# change the model into its evaluation setting
sheep_classifier.eval()

# concatenate along new first dimension
x_test_stack = None

# x_test_stack
for i in enumerate(x_test):
    if x_test_stack == None:
        x_test_stack = torch.unsqueeze(i[1], dim=0)
    else:
        x_test_stack = torch.cat((x_test_stack, torch.unsqueeze(i[1], dim=0)), dim=0)

# y_test_stack
y_test_stack = None

# populate the y_test_stack variable
for z in enumerate(y_test):
    if y_test_stack == None:
        y_test_stack = torch.unsqueeze(z[1], dim=0)
    else:
        y_test_stack = torch.cat((y_test_stack, torch.unsqueeze(z[1], dim=0)), dim=0)
model_preds = sheep_classifier(x_test_stack)  # input testing data
model_preds = np.argmax(model_preds.detach().numpy(), axis=1)
accuracy = metrics.accuracy_score(y_test_stack, model_preds)  # check accuracy
print(f'Overall testing accuracy is {round((accuracy * 100), 2)}% (2 d.p)')
# a function to present images of sheep within the testing set as well as 
# the model's prediction and the correct label
def result_presentation(x_test, y_test, prediction, labels):
    
    ran = random.sample(range(0, x_test.shape[0]), 6)
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(2, 3, figsize=(10,10))
    
    ax[0, 0].imshow(x_test[ran[0]].numpy())
    ax[0, 0].set_title("Predicted: " + labels[prediction[ran[0]]] + 
                       "\nTruth: " + labels[y_test[ran[0]]], fontsize=15)

    ax[0, 1].imshow(x_test[ran[1]].numpy())
    ax[0, 1].set_title("Predicted: " + labels[prediction[ran[1]]] + 
                       "\nTruth: " + labels[y_test[ran[1]]], fontsize=15)
    
    ax[0, 2].imshow(x_test[ran[2]].numpy())
    ax[0, 2].set_title("Predicted: " + labels[prediction[ran[2]]] + 
                       "\nTruth: " + labels[y_test[ran[2]]], fontsize=15)
    
    ax[1, 0].imshow(x_test[ran[3]].numpy())
    ax[1, 0].set_title("Predicted: " + labels[prediction[ran[3]]] + 
                       "\nTruth: " + labels[y_test[ran[3]]], fontsize=15)
    
    ax[1, 1].imshow(x_test[ran[4]].numpy())
    ax[1, 1].set_title("Predicted: " + labels[prediction[ran[4]]] + 
                       "\nTruth: " + labels[y_test[ran[4]]], fontsize=15)
    
    ax[1, 2].imshow(x_test[ran[5]].numpy())
    ax[1, 2].set_title("Predicted: " + labels[prediction[ran[5]]] + 
                       "\nTruth: " + labels[y_test[ran[5]]], fontsize=15)
    
    f.tight_layout(pad=1.5)
    
    return

# call the function
result_presentation(x_test_stack, y_test_stack, model_preds, unique_labels)
# define a function that calculates the accuracy of each sheep breed
def accuracy_by_breed(y_test, predictions, label):
    
    correct = 0
    labels = 0
    for truth, predicted in zip(y_test.tolist(), predictions):
        #print(truth, predicted)
        if truth == label:
            labels += 1
        
        if truth == label and predicted == label:
            correct += 1
                                    
    return (correct / labels) * 100 

# determine the accuracy of testing predictions by sheep breed
label_0 = accuracy_by_breed(y_test_stack, model_preds, 0)
label_1 = accuracy_by_breed(y_test_stack, model_preds, 1)
label_2 = accuracy_by_breed(y_test_stack, model_preds, 2)
label_3 = accuracy_by_breed(y_test_stack, model_preds, 3)

# plot as a barchart
sns.set(rc={'figure.figsize': (45.0, 20.0)})
sns.set_context("notebook", font_scale=4.5, rc={"lines.linewidth": 0.5})
ax = sns.barplot(x=unique_labels,
                 y=[label_0, label_1, label_2, label_3])
ax.set_ylabel('Classification accuracy (%)', labelpad=40, fontsize=65)
ax.set_xlabel('Sheep Breed', labelpad=40, fontsize=65)
plt.title("A Barplot comparing the model classification \naccuracy by sheep breed", fontsize=100)