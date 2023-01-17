# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot # plotly offline mode

init_notebook_mode(connected=True) 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# for image read i used opencv to resize and east read png files 

data1 = cv2.imread("../input/cell_images/cell_images/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png") #parasitized malari cell path and call one image

data1 = cv2.resize(data1,(160,160)) 

plt.subplot(1,2,1)

plt.imshow(data1)

plt.axis("off")

plt.title("Parasitized")

data1 = cv2.imread("../input/cell_images/cell_images/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png") #uninfected malari cell path and call one image

data1 = cv2.resize(data1,(160,160))

plt.subplot(1,2,2)

plt.imshow(data1)

plt.axis("off")

plt.title("Uninfected")

plt.show()
x = list() #create x data

y = list() # create y data

for i in os.listdir("../input/cell_images/cell_images/Parasitized"): #read all parasitized data 

    if ".png" in i: #this if block for only read .png files

        path = "../input/cell_images/cell_images/Parasitized/"+i # create path

        img = plt.imread(path) # and read created path

        img = cv2.resize(img,(40,40)) # resize image for lower processing power

        x.append(img) # append image to x data

        y.append(1) 

for i in os.listdir("../input/cell_images/cell_images/Uninfected/"):

    if ".png" in i:

        path = "../input/cell_images/cell_images/Uninfected/"+i

        img = plt.imread(path)

        img = cv2.resize(img,(40,40))

        x.append(img)

        y.append(0)

x = np.array(x)  
# create 4 subplots and plot 4 random image 

plt.subplot(1,4,1)

plt.imshow(x[2000]) # image 1

plt.title(y[2000])

plt.axis("off")

plt.subplot(1,4,2) # image 2

plt.imshow(x[22000])

plt.title(y[22000])

plt.axis("off")

plt.subplot(1,4,3) #image 3

plt.imshow(x[20000])

plt.title(y[20000])

plt.axis("off")

plt.subplot(1,4,4) #image 4 

plt.imshow(x[200])

plt.title(y[200])

plt.axis("off")

plt.show()
#reshapeing data

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])

x = x**8
# i used sklearn modul for splitting process

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
img1 =x_train[970].reshape(40,40,3) #reshape for normal image shape

img2 =x_train[100].reshape(40,40,3)

plt.subplot(1,2,1)

plt.imshow(img1)

plt.axis("off")

plt.title(y_train[900])

plt.subplot(1,2,2)

plt.imshow(img2)

plt.axis("off")

plt.title(y_train[100])

plt.show()
#import LogisticRegression and fit with out datas

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="lbfgs")

lr.fit(x_train,y_train)
# Test my Logistic Regression Model

print("Logistic Regression Accuracy : {0:.2f}%".format(100 * lr.score(x_test,y_test)))
#import modules i will use

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

# build our model

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1])) # firt hidden layer 

    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # last layer

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier # return our model
model = build_classifier()

model.summary()
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

history = classifier.fit(x_train,y_train)
history.history['acc']

history.history['loss']

x_ = np.array(range(len(history.history['loss'])))

trace1 = go.Scatter(

        x = x_,

        y = history.history['loss'],

        mode = "lines",

        marker = dict(color = "rgba(0,255,0,0.9)"),

        text = "Loss"

)

trace2 = go.Scatter(

        x = x_,

        y = history.history['acc'],

        mode = "lines",

        marker = dict(color = "rgba(0,0,255,0.9)"),

        text = "Accuracy"

)

data = [trace1,trace2]

layout = dict(title = "Training Accuracy and Loss")

fig = dict(data = data,layout=layout)

iplot(fig)
classifier.score(x_test,y_test)