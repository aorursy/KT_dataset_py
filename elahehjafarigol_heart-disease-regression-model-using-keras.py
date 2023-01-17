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
#Import libraries

import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt



%matplotlib inline
#Load spreadsheet

file = pd.read_csv('../input/framingham_heart_disease.csv')



#Exploratory data analysis

print (file.shape)

print (file.info())
#Missing value report

Missing_Value_Percentage = 100 - ((file.count().sort_values()/len(file))*100)

print ("The percentage of missing values in each column is:")

Missing_Value = pd.DataFrame(Missing_Value_Percentage)

Missing_Value.columns =  ['Missing value report']

Missing_Value
data = pd.DataFrame (file)



fill_feat = ["glucose", "education", "BPMeds", "totChol", "cigsPerDay", "BMI", "heartRate"]

for i in fill_feat: 

    data[i].fillna(np.mean(data[i]),inplace=True)
#Normalization using Sklearn

Data = np.asarray(data[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',

       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose']])

Target = data['TenYearCHD']



#Scaling the data before training

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler





Data = preprocessing.StandardScaler().fit(Data).transform(Data)

Data = pd.DataFrame (Data)

Data.columns = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',

       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',

       'diaBP', 'BMI', 'heartRate', 'glucose']

#print (Data.head())
Target.columns = ["label"]
#Correlation map of the original data scaled

def Corr_heatmap (data):

    sb.set(style="whitegrid",font='sans-serif', font_scale=1.3)

    plt.figure(figsize=(15, 15))

    plt.title('Correlation Matrix')

    plot = sb.heatmap(data.corr(), annot=True,cmap= 'coolwarm', fmt='.2f')

    plt.show

Corr_heatmap (data)    
#4) Recursive feature elimination with cross validation and SVM classification

from sklearn.feature_selection import RFECV

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE

# Create the RFE object and rank each pixel





clf_rf = RandomForestClassifier() 



rfecv = RFECV(estimator=clf_rf, step=1, cv=5, scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(Data, Target)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', Data.columns[rfecv.support_])
# Plot number of features VS. cross-validation scores

import matplotlib.pyplot as plt

plt.figure()

plt.xlabel("Number of features selected", fontsize = 12)

plt.ylabel("Cross-Validation score", fontsize = 12)



plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
droplist = ['BPMeds', 'prevalentHyp','diabetes']

Data = Data.drop(droplist, axis = 1 )

Data.head()

clf_rf = RandomForestClassifier()      

clr_rf = clf_rf.fit (Data, Target)

importances = clr_rf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_], axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(Data.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(1, figsize=(10, 10))

plt.title("Feature importances", fontsize = 16)

plt.bar(range(Data.shape[1]), importances[indices], color="darkred", yerr=std[indices], align="center")

plt.xticks(range(Data.shape[1]), Data.columns[indices],rotation=90)

plt.xlim([-1, Data.shape[1]])

plt.show()
#Import Tenserflow

import tensorflow as tf
#Define the model

lr_model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=1, input_shape=[1]),

])

#unit = the number of neurons

#input_shape tells the shape of input in the first layer.
lr_model.compile(optimizer='adam', loss='mean_squared_error')
#Divide data into training and test sets

from sklearn.model_selection import train_test_split



X = Data["BMI"]

Y = Data["sysBP"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1) 
#training_model = lg_model.fit (X_train, y_train, epochs = 500)

#training_model = lg_model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

training_model = lr_model.fit(X_train, y_train, epochs=500, verbose=1)
import matplotlib.pyplot as plt

with plt.xkcd():

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(np.arange(500), training_model.history['loss'], 'b-', label='loss')

    xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('loss')

    plt.show()
y_pred = lr_model.predict(X_test)

print (y_pred)
lg_model = tf.keras.Sequential([

    tf.keras.layers.Dense(units=1, input_shape=[12], activation='sigmoid')])
lg_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
#Divide data into training and test sets

from sklearn.model_selection import train_test_split



X = Data

Y = Target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 1) 
training_model = lg_model.fit(X_train, y_train, epochs=500)
with plt.xkcd():

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(np.arange(500), training_model.history['loss'], 'b-', label='loss')

    xlab, ylab = ax.set_xlabel('epoch'), ax.set_ylabel('loss')

    plt.show()
y_pred = lg_model.predict(X_test)

print (y_pred)