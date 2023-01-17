# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
from keras.applications.vgg19 import VGG19

from keras.utils import to_categorical

import cv2

import numpy as np

from keras.layers import Dense, Flatten

from glob import glob
X = np.load('../input/X.npy') # images

Y = np.load('../input/Y.npy') # labels associated to images (0 = no IDC, 1 = IDC)
imgsize = 64

plt.subplot(1,4,1)

plt.imshow(cv2.cvtColor(X[1], cv2.COLOR_BGR2RGB))

plt.axis("on")



plt.subplot(1,4,2)

plt.imshow(cv2.cvtColor(X[2], cv2.COLOR_BGR2RGB))

plt.axis("on")



plt.subplot(1,4,3)

plt.imshow(cv2.cvtColor(X[3], cv2.COLOR_BGR2RGB))

plt.axis("on")



plt.subplot(1,4,4)

plt.imshow(cv2.cvtColor(X[4], cv2.COLOR_BGR2RGB))

plt.axis("on")
print("X shape: ", X.shape)

print("Y shape: ", Y.shape)
#Normalization

X = X / 255.0

print("X Shape:",X.shape)
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state=2)

numberoftrain = xtrain.shape[0]

numberoftest = xtest.shape[0]

xtrain.shape
#Reshape Xtrain & Xtest



xtrain = xtrain.reshape(numberoftrain,xtrain.shape[1]*xtrain.shape[2]*xtrain.shape[3])

xtest = xtest.reshape(numberoftest,xtest.shape[1]*xtest.shape[2]*xtest.shape[3])

print("X Train: ",xtrain.shape)

print("X Test: ",xtest.shape)

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library



def buildclassifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = xtrain.shape[1]))

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier
classifier = KerasClassifier(build_fn = buildclassifier, epochs = 200)

accuracies = cross_val_score(estimator = classifier, X = xtrain, y = ytrain, cv = 6)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(xtrain,ytrain) #learning
#prediciton

dtscore = DTC.score(xtest,ytest)

print("Decision Tree Score: ",DTC.score(xtest,ytest))
#Random Forest



from sklearn.ensemble import RandomForestClassifier

RFC= RandomForestClassifier(n_estimators = 100, random_state=42) #n_estimator = DT

RFC.fit(xtrain,ytrain) # learning

rfsc=RFC.score(xtest,ytest)

print("Random Forest Score: ",RFC.score(xtest,ytest))
#SVM with Sklearn



from sklearn.svm import SVC



SVM = SVC(random_state=42)

SVM.fit(xtrain,ytrain)  #learning 

#SVM Test 

svmsc = SVM.score(xtest,ytest)

print ("SVM Accuracy:", SVM.score(xtest,ytest))
from sklearn.model_selection import train_test_split



xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, random_state=2)

numberoftrain = xtrain.shape[0]

numberoftest = xtest.shape[0]

xtrain.shape
numberOfClass = 2





ytrain = to_categorical(ytrain, numberOfClass)

ytest = to_categorical(ytest, numberOfClass)



input_shape = xtrain.shape[1:]
def resize_img(img):

    numberOfImage = img.shape[0]

    new_array = np.zeros((numberOfImage, 64,64,3))

    for i in range(numberOfImage):

        new_array[i] = cv2.resize(img[i,:,:,:],(64,64))

    return new_array



xtrain = resize_img(xtrain)

xtest = resize_img(xtest)

print("increased dim x_train: ",xtrain.shape)


vgg = VGG19(include_top = False, weights = "imagenet", input_shape = (64,64,3))



print(vgg.summary())
vgg_layer_list = vgg.layers

#print(vgg_layer_list)
model = Sequential()

for layer in vgg_layer_list:

    model.add(layer)

    

print(model.summary())
for layer in model.layers:

    layer.trainable = False



# fully con layers

model.add(Flatten())

model.add(Dense(128))

model.add(Dense(128))

model.add(Dense(numberOfClass, activation= "sigmoid"))



print(model.summary())
model.compile(loss = "binary_crossentropy",

              optimizer = "rmsprop",

              metrics = ["accuracy"])
#validation_split = 0.2, epochs = 100, batch_size = 1000

hist = model.fit(xtrain, ytrain, validation_split = 0.3, epochs = 100, batch_size = 1000)
plt.title('vgg19 - Loss')

plt.plot(hist.history["loss"], label = "train loss")

plt.plot(hist.history["val_loss"], label = "val loss")

plt.legend()

plt.show()



plt.figure()

plt.title('vgg19 - Accuracy')

plt.plot(hist.history["acc"], label = "train acc")

plt.plot(hist.history["val_acc"], label = "val acc")

plt.legend()

plt.show()
acc1= np.mean(hist.history["acc"])
scoresf1=[mean,dtscore,rfsc,svmsc,acc1]

#create traces

AlgorthmsName=["Artificial Neural Network","Decision Tree",

                "Random Forest","Support Vector Machine",'VGG19-Transfer Learning-']



trace1 = go.Scatter(

    x = AlgorthmsName,

    y= scoresf1,

    name='Algortms Name',

    marker =dict(color='rgba(225,126,0,0.5)',

               line =dict(color='rgb(0,0,0)',width=2)),

                text=AlgorthmsName

)

data = [trace1]



layout = go.Layout(barmode = "group", 

                  xaxis= dict(title= 'ML Algorithms',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Prediction Scores(F1)',ticklen= 5,zeroline= False))

fig = go.Figure(data = data, layout = layout)

iplot(fig)