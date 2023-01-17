# Credit Card Fraud Detection using Neural Networks
# By - Omkar Sabnis: 22-05-2018

# IMPORTING REQUIRED MODULES
import numpy as np
print(np.__version__)
import pandas as pd
print(pd.__version__)
import matplotlib.pyplot as plt
import seaborn as sns
print(sns.__version__)
import itertools
import warnings
warnings.filterwarnings("ignore")
# NEURAL NETWORKS MODULES
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# KERAS MODULES
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import Callback
import keras.backend as kb
import tensorflow as tf
print(tf.__version__)
# READING THE DATASET
dataset = pd.read_csv("../input/creditcard.csv")
print("Few Entries: ")
print(dataset.head())
print("Dataset Shape: ", dataset.shape)
print("Maximum Transaction Value: ", np.max(dataset.Amount))
print("Minimum Transaction Value: ", np.min(dataset.Amount))
# PLOTTING A FEW GRAPHS
color = {1:'blue',0:'yellow'}
fraudlist = dataset[dataset.Class == 1]
notfraudlist = dataset[dataset.Class == 0]
fig,axes = plt.subplots(1,2)
axes[0].scatter(list(range(1,fraudlist.shape[0]+1)),fraudlist.Amount,color='blue')
axes[1].scatter(list(range(1,notfraudlist.shape[0]+1)),notfraudlist.Amount,color='yellow')
plt.show()
# SETTING UP THE TRAINING AND TESTING SETS
x = dataset.loc[:,dataset.columns.tolist()[1:30]]
x = x.as_matrix()
y = dataset.loc[:,'Class']
y = y.as_matrix()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
print("Elements in the training set:" , np.bincount(y_train))
print("Elements in the testing set:" , np.bincount(y_test))
print(x_train)
# FUNCTION FOR TRAINING THE MODEL
def trainmodel(model):
    model.fit(x_train,y_train)
# FUNCTION TO MAKE PREDICTIONS
def predictmodel(model):
    y_pred = model.predict(x_test)
    f,t,thresholds = metrics.roc_curve(y_test,y_pred)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print("Score:", metrics.auc(f,t))
    print("Classification report:")
    print(metrics.classification_report(y_test,y_pred))
    print("Confusion Matrix:")
    print(cm)
# FUNCTION TO MAKE PREDICTIONS
def predictmodeln(model):
    y_pred = model.predict_classes(x_test)
    f,t,thresholds = metrics.roc_curve(y_test,y_pred)
    cm = metrics.confusion_matrix(y_test,y_pred)
    print("Score:", metrics.auc(f,t))
    print("Classification report:")
    print(metrics.classification_report(y_test,y_pred))
    print("Confusion Matrix:")
    print(cm)
# DEFINING THE NEURAL NETWORK
model = Sequential()
model.add(Dense(256,activation='sigmoid',input_dim=29))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
# CHECKING THE OUTPUT OF THE DEFINED NETWORK
model.fit(x_train,y_train,epochs=5)
print(predictmodeln(model))
# REPLICATION OF THE SMALLEST CLASS
fraudlist = x_train[y_train==1]
y_fraudlist = y_train[y_train==1]
print(x_train.shape)

for _ in range(5):
    copy_fraudlist = np.copy(fraudlist)
    y_fraud_copy = np.copy(y_fraudlist)
    x_train = np.concatenate((x_train,copy_fraudlist))
    y_train = np.concatenate((y_train,y_fraud_copy))

permut = np.random.permutation(x_train.shape[0])
x_train = x_train[permut]
y_train = y_train[permut]
print(x_train)
# REDEFINE THE NEURAL NETWORK
model = Sequential()
model.add(Dense(256,activation='sigmoid',input_dim=29))
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(64,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# INCREASING ACCURACY BY MULTIPLE PASSES
model.fit(x_train,y_train,epochs=5)
print(predictmodeln(model))

model.fit(x_train,y_train,epochs=5)
print(predictmodeln(model))
# DECISION TREE CLASSIFIER
dtc = DecisionTreeClassifier()
trainmodel(dtc)
predictmodel(dtc)
# NAIVE BAYES CLASSIFIER
gnb = GaussianNB()
trainmodel(gnb)
predictmodel(gnb)