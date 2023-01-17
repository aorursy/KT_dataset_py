import pandas as pd

import numpy as np

import seaborn as sns



# read data into dataset variable

data = pd.read_csv("../input/Dataset_spine.csv")



# Drop the unnamed column in place (not a copy of the original)#

data.drop('Unnamed: 13', axis=1, inplace=True)



# Concatenate the original df with the dummy variables

data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)



# Drop unnecessary label column in place. 

data.drop(['Class_att','Normal'], axis=1, inplace=True)
data.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius', 

                'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope', 

                'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']



corr = data.corr()



# Set up the matplot figure

f, ax = plt.subplots(figsize=(12,9))



#Draw the heatmap using seaborn

sns.heatmap(corr, cmap='inferno', annot=True)
data.describe()
from pylab import *

import copy

outlier = data[["Spondylolisthesis Degree", "Outcome"]]

#print(outlier[outlier >200])

abspond = outlier[outlier["Spondylolisthesis Degree"]>15]

print("1= Abnormal, 0=Normal\n",abspond["Outcome"].value_counts())
#   Dropping Outlier

data = data.drop(115,0)

colr = copy.copy(data["Outcome"])

co = colr.map({1:0.44, 0:0.83})



#   Plot scatter

plt.scatter(data["Cervical Tilt"], data["Spondylolisthesis Degree"], c=co, cmap=plt.cm.RdYlGn)

plt.xlabel("Cervical Tilt")

plt.ylabel("Spondylolisthesis Degree")



colors=[ 'c', 'y', 'm',]

ab =data["Outcome"].where(data["Outcome"]==1)

no = data["Outcome"].where(data["Outcome"]==0)

plt.show()

# UNFINISHED ----- OBJECTIVE: Color visual by Outcome - 0 for green, 1 for Red (example)
#   Create the training dataset

training = data.drop('Outcome', axis=1)

testing = data['Outcome']
#   Import necessary ML packages

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report



#   Split into training/testing datasets using Train_test_split

X_train, X_test, y_train, y_test = train_test_split(training, testing, test_size=0.33, random_state=22, stratify=testing)
import numpy as np



# convert to numpy.ndarray and dtype=float64 for optimal

array_train = np.asarray(training)

array_test = np.asarray(testing)

print(array_train.shape)

print(array_test.shape)



#   Convert each pandas DataFrame object into a numpy array object. 

array_XTrain, array_XTest, array_ytrain, array_ytest = np.asarray(X_train), np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)
#    Import Necessary Packages

from sklearn import svm

from sklearn.metrics import accuracy_score



#   Instantiate the classifier

clf = svm.SVC(kernel='linear')



#   Fit the model to the training data

clf.fit(array_XTrain, array_ytrain)



#   Generate a prediction and store it in 'pred'

pred = clf.predict(array_XTest)



#   Print the accuracy score/percent correct

svmscore = accuracy_score(array_ytest, pred)

print("Support Vector Machines are ", svmscore*100, "accurate")

from keras.models import Sequential

from keras.layers import Dense, Activation

import keras
print(array_XTrain.shape)

print(array_ytrain.shape)
#  Define our model

model = Sequential()

model.add(Dense(32, activation='tanh', input_dim=12))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



one_hot_labels = keras.utils.to_categorical(array_ytrain, num_classes=10)



history = model.fit(array_XTrain, one_hot_labels,epochs=1000, batch_size=30)

weights = model.layers[0].get_weights()[0]

biases = model.layers[0].get_weights()[1]
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot(history.history['loss'])

plt.xlabel("Epochs (Batches)")

plt.ylabel("Loss")

plt.title("Training an Artificial Neural Net")