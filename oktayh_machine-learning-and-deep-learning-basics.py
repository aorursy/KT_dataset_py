# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import plotly.offline as py

py.init_notebook_mode(connected=True)

import matplotlib.pyplot as plt # side-stepping mpl backend

import matplotlib.gridspec as gridspec # subplots

import mpld3 as mpl

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from sklearn.model_selection import cross_validate

#Import models from scikit learn module:

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics

from pandas.plotting import scatter_matrix

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
data.head(20)
data.columns
data.isnull().sum()
data.drop(['Unnamed: 32',"id"], axis=1, inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(['diagnosis'], axis=1)

sns.countplot(data['diagnosis'],label="Count")
scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(x_data)

type(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
logr = LogisticRegression()

logr.fit(x_train,y_train)

y_predict_lr = logr.predict(x_test)

acc_log = metrics.accuracy_score(y_predict_lr,y_test)*100

print('LogisticRegression accuracy(in %):', acc_log)
sv = SVC() #select the algorithm

sv.fit(x_train,y_train) # we train the algorithm with the training data and the training output

y_predict_svm = sv.predict(x_test) #now we pass the testing data to the trained algorithm

acc_svm = metrics.accuracy_score(y_predict_svm,y_test)*100

print('SVM accuracy(in %):', acc_svm)
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

y_predict_dt = dt.predict(x_test)

acc_dt = metrics.accuracy_score(y_predict_dt,y_test)*100

print('Decision Tree accuracy(in %):', acc_dt)
from sklearn.metrics import confusion_matrix

rf = RandomForestClassifier(n_estimators = 40)

rf.fit(x_train,y_train)

rf_pred = rf.predict(x_test)

acc_rfc = metrics.accuracy_score(rf_pred,y_test)*100

print("RandomForestClassifier accuracy(in %):", acc_rfc)
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB() 

gnb.fit(x_train, y_train) 

gnb_pred = gnb.predict(x_test)

acc_gnb = metrics.accuracy_score(gnb_pred,y_test)*100

print("Gaussian Naive Bayes model accuracy(in %):", acc_gnb)
knc = KNeighborsClassifier(n_neighbors=5)

knc.fit(x_train,y_train)

y_predict_knn = knc.predict(x_test)

acc_knn = metrics.accuracy_score(y_predict_knn,y_test)*100

print("KNN model accuracy(in %):", acc_knn)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Machines','Decision Tree',

              'Random Forest Classifier','Gaussian Naive Bayes',

              'K-Nearest Neighbours'],

    'Score': [acc_log,acc_svm,acc_dt,acc_rfc,acc_gnb,acc_knn]})

models
vecTest = y_test.reshape((-1))

result = pd.DataFrame({'ActualValues':vecTest,'LogisticRegression':y_predict_lr,'SVM':y_predict_svm,'DecisionTree':y_predict_dt,'KNN':y_predict_knn,'GausianBayes':gnb_pred,'RandomForest':rf_pred})

result
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
plt.figure(figsize=(10,15))

# Logistic Regression

plt.subplot(4,2,1)

cm = confusion_matrix(y_test, y_predict_lr) 

cm_lr = pd.DataFrame(cm)

sns.heatmap(cm_lr,cmap="Oranges",annot = True)

plt.title('Logistic Regression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_lr)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

#Support Machine Vector

plt.subplot(4,2,2)

cm = confusion_matrix(y_test, y_predict_svm) 

cm_svm = pd.DataFrame(cm)

sns.heatmap(cm_svm,cmap="Blues",annot = True)

plt.title('SVM \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_svm)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

#Decision Tree

plt.subplot(4,2,3)

cm = confusion_matrix(y_test, y_predict_dt) 

cm_dt = pd.DataFrame(cm)

sns.heatmap(cm_dt,cmap="Purples",annot = True)

plt.title('Decision Tree \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_dt)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

#KNN

plt.subplot(4,2,4)

cm = confusion_matrix(y_test, y_predict_knn) 

cm_knn = pd.DataFrame(cm)

sns.heatmap(cm_dt,cmap="Greens",annot = True)

plt.title('KNN \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_predict_knn)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

#Gausian Bayes

plt.subplot(4,2,5)

cm = confusion_matrix(y_test, gnb_pred) 

cm_gb = pd.DataFrame(cm)

sns.heatmap(cm_gb,cmap="Greys",annot = True)

plt.title('Gausian Bayes \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, gnb_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')

#Random Forest Classifier

plt.subplot(4,2,6)

cm = confusion_matrix(y_test, rf_pred) 

cm_rf = pd.DataFrame(cm)

sns.heatmap(cm_rf,cmap="Reds",annot = True)

plt.title('Random Forest \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, rf_pred)))

plt.ylabel('True label')

plt.xlabel('Predicted label')



plt.tight_layout()

plt.show()
from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score
print("Logistic Regression Precision score: {}".format(precision_score(y_test,y_predict_lr)))

print("Logistic Regression Recall score: {}".format(recall_score(y_test,y_predict_lr)))

print("Logistic Regression F1 Score: {}".format(f1_score(y_test,y_predict_lr)))
print("SVM Precision score: {}".format(precision_score(y_test,y_predict_svm)))

print("SVM Recall score: {}".format(recall_score(y_test,y_predict_svm)))

print("SVM F1 Score: {}".format(f1_score(y_test,y_predict_svm)))
print("Decision Tree Precision score: {}".format(precision_score(y_test,y_predict_dt)))

print("Decision Tree Recall score: {}".format(recall_score(y_test,y_predict_dt)))

print("Decision Tree F1 Score: {}".format(f1_score(y_test,y_predict_dt)))
print("KNN Precision score: {}".format(precision_score(y_test,y_predict_knn)))

print("KNN Recall score: {}".format(recall_score(y_test,y_predict_knn)))

print("KNN F1 Score: {}".format(f1_score(y_test,y_predict_knn)))
print("GNB Precision score: {}".format(precision_score(y_test,gnb_pred)))

print("GNB Recall score: {}".format(recall_score(y_test,gnb_pred)))

print("GNB F1 Score: {}".format(f1_score(y_test,gnb_pred)))
print("Random Forest Precision score: {}".format(precision_score(y_test,rf_pred)))

print("Random Forest Recall score: {}".format(recall_score(y_test,rf_pred)))

print("Random Forest F1 Score: {}".format(f1_score(y_test,rf_pred)))


plt.bar(models['Model'],models['Score'])

plt.xticks(rotation=90)
from sklearn.model_selection import cross_val_score
cvs_lr = cross_val_score(LogisticRegression(),x,y,cv=5)

max_lr = np.amax(cvs_lr)

cvs_svm = cross_val_score(SVC(),x,y,cv=5)

max_svm = np.amax(cvs_svm)

cvs_gnb = cross_val_score(GaussianNB(),x,y,cv=5)

max_gnb = np.amax(cvs_gnb)

cvs_rf = cross_val_score(RandomForestClassifier(n_estimators = 40),x,y,cv=5)

max_rf = np.amax(cvs_rf)



scores = pd.DataFrame({

    

    'Name' : {0:'LogisticRegression',1:'Support Vector Machines',2:'Gaussian Naive Bayes',3:'Random Forest Classifier'} ,  

    'Cross_val' : {0:cvs_lr,1:cvs_svm,2:cvs_gnb,3:cvs_rf}

    

})

scores
print('Max Cross Validation Score of Logistic Regression {}'.format(max_lr))

print('Max Cross Validation Score of Support Machine Vector {}'.format(max_svm))

print('Max Cross Validation Score of Gaussian Naive Bayes {}'.format(max_gnb))

print('Max Cross Validation Score of Random Forest Classifier {}'.format(max_rf))



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(x_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20)

X_val, X_test, Y_val, Y_test = train_test_split(X_validation, Y_validation, test_size=0.40)

import tensorflow as tf

from tensorflow.keras import layers

from keras.regularizers import l2
Y_train = Y_train.reshape((-1,1))
Y_train.shape
learning_rate = 0.001

epochs = 200

batch_size = 256



model = tf.keras.models.Sequential()

    

model.add(tf.keras.layers.Dense(units=8,input_shape=(X_train.shape[1],),activation= tf.nn.relu,

                                    kernel_regularizer=l2(0.001),name='Input'))

model.add(tf.keras.layers.Dense(units = 16,activation = tf.nn.relu,

                                    kernel_regularizer=l2(0.001),name='HiddenLayer-1'))

model.add(tf.keras.layers.Dense(units = 16,activation = tf.nn.relu,

                                    kernel_regularizer=l2(0.001),name='HiddenLayer-2'))

model.add(tf.keras.layers.Dense(units = 32,activation = tf.nn.relu,

                                    kernel_regularizer=l2(0.001),name='HiddenLayer-3'))

model.add(tf.keras.layers.Dense(Y_train.shape[1],activation=tf.nn.sigmoid,name='Output'))



model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),                                                   

                loss=tf.keras.losses.BinaryCrossentropy(),

                metrics=['accuracy'])
history = model.fit(x=x, y=y,validation_data = (X_test,Y_test),batch_size=batch_size,epochs=epochs)
model.evaluate(X_test, Y_test, verbose=1,batch_size=batch_size)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper right')

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='lower right')

plt.show()
pred = model.predict(X_test)

pred_values = pd.DataFrame(pred,columns = ['label'])

pred_values['Model_Prediction'] = (pred_values.label >0.5).astype('int')

pred_values['Actual_values'] = Y_test

pred_values.drop('label', axis = 1)
from tensorflow.keras.utils import plot_model

plot_model(model)
# sns.pairplot(data[0:10], hue='diagnosis',palette='cubehelix',kind = 'reg')
clear