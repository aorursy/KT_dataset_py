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
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head(2)
import seaborn as sns
sns.heatmap(df.corr(),annot=True)
#we can see 'Glucose' , 'BMI' and 'Age' are highly co-related with our data
x = df.iloc[:,:8]
y =df.iloc[:,8]
#rescaling our data for better calculatons
from sklearn import preprocessing
x = preprocessing.scale(x)
x #preprocessed data

sns.countplot(x = 'Outcome',data=df,palette='hls')
#now import LogisticRegression library from sklearn
from sklearn.linear_model import LogisticRegression
#making a logistic regression objext
logit_reg = LogisticRegression()




#splitting data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
x_train
y_train
logit_reg.fit(x_train,y_train)
#fitting our model
pred_y = logit_reg.predict(x_test)
#predicting on our test data

#checking accuracy of logistic regreesion model
print('score of model on test data is ',logit_reg.score(x_test,y_test))
print('score of model on train data is',logit_reg.score(x_train,y_train))
from sklearn.metrics import confusion_matrix , classification_report
confusion_matrix = confusion_matrix(y_test, pred_y)

print('classification report is ',classification_report(y_test,pred_y))
#visualising confusion matrix
sns.heatmap(confusion_matrix,annot= True)
#checking statistical method
import statsmodels.api as sm
logit_model=sm.Logit(y,x)
result=logit_model.fit()
print(result.summary())

# we should also check how many k neaighbours we should have for our model for best results
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_range = range(1,26)
scores = {}
score_list = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test,y_pred)
    score_list.append(metrics.accuracy_score(y_test,y_pred))
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(k_range,score_list)
plt.xlabel('value of k for knn')
plt.ylabel('test set accuracy')
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
print('accuracy on test data is',knn.score(x_test,y_test))
print('accuracy on train data is',knn.score(x_train,y_train))
print('model is generalising well')

import keras
from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(5,activation = 'relu',input_dim=8))
classifier.add(Dense(5,activation = 'relu'))
#output layer
classifier.add(Dense(1,activation='sigmoid'))
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])
history =  classifier.fit(x_train,y_train,batch_size=10,epochs=20,validation_split=0.2)
eval_model_train=classifier.evaluate(x_train, y_train)
print('training set accuracy',eval_model_train)
eval_model_test = classifier.evaluate(x_test,y_test)
print('testing set accuracy',eval_model_test)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
from sklearn.model_selection import GridSearchCV,KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
def create_model():
    model = Sequential()
    model.add(Dense(8,input_dim = 8,kernel_initializer = 'normal',activation='relu'))
    model.add(Dense(4,kernel_initializer = 'normal',activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    
    adam = Adam(lr=0.01)
    model.compile(loss = 'binary_crossentropy',optimizer=adam,metrics = ['accuracy'])
    
    return model
#create model
model = KerasClassifier(build_fn=create_model,verbose = 0)

#define grid search parameters
batch_size = [10,20,40]
epochs = [10,50,100]

param_grid = dict(batch_size=batch_size,epochs=epochs)
grid = GridSearchCV(estimator = model,param_grid = param_grid,cv = KFold(),verbose = 10)
grid_result = grid.fit(x_train,y_train)
print(grid_result.best_score_,grid_result.best_params_)
#turning learning rate and drop out rate
from keras.layers import Dropout

# Defining the model

def create_model(learning_rate,dropout_rate):
    model = Sequential()
    model.add(Dense(8,input_dim = 8,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4,input_dim = 8,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = Adam(lr = learning_rate)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)

# Define the grid search parameters

learning_rate = [0.001,0.01,0.1]
dropout_rate = [0.0,0.1,0.2]

# Make a dictionary of the grid search parameters

param_grids = dict(learning_rate = learning_rate,dropout_rate = dropout_rate)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x_train,y_train)

print(grid_result.best_score_,grid_result.best_params_)
#best results are at drop out rate of 0.2 and learning rate of 0.01
def create_model(activation_function,init):
    model = Sequential()
    model.add(Dense(8,input_dim = 8,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.1))
    model.add(Dense(4,input_dim = 8,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)

# Define the grid search parameters

activation_function = ['softmax','relu','tanh','linear']
init = ['uniform','normal','zero']

# Make a dictionary of the grid search parameters

param_grids = dict(activation_function = activation_function,init = init)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 2)
grid_result = grid.fit(x_train,y_train)
print(grid_result.best_score_,grid_result.best_params_)
def create_model(neuron1,neuron2):
    model = Sequential()
    model.add(Dense(neuron1,input_dim = 8,kernel_initializer = 'uniform',activation = 'tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = 'uniform',activation = 'tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)

# Define the grid search parameters

neuron1 = [4,8,16]
neuron2 = [2,4,8]

# Make a dictionary of the grid search parameters

param_grids = dict(neuron1 = neuron1,neuron2 = neuron2)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x_train,y_train)
print(grid_result.best_score_,grid_result.best_params_)
from sklearn.metrics import classification_report, accuracy_score

# Defining the model

def create_model():
    model = Sequential()
    model.add(Dense(16,input_dim = 8,kernel_initializer = 'uniform',activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(4,input_dim = 16,kernel_initializer = 'uniform',activation = 'tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation = 'sigmoid'))
    
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)

# Fitting the model

model.fit(x_train,y_train)
y_pred_tuned =model.predict(x_test)
print('classification report is',classification_report(y_pred_tuned,y_test))
print(accuracy_score(y_pred_tuned,y_test))
