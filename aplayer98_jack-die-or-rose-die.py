import sys #system

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib #graph plotting

import scipy as sp #scientific calculation

import sklearn #machine learning models

import xgboost as xgb # for xgBoost learning model



from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt #visualization

import seaborn as sns



import warnings

warnings.filterwarnings('ignore') #ignore warnings



import os

print(os.listdir("../input")) #show stuffs under input

# Any results you write to the current directory are saved as output.
#time to read the data first

raw_data_train = pd.read_csv('../input/train.csv')

raw_data_test = pd.read_csv('../input/test.csv')

raw_data = [raw_data_train,raw_data_test]

def print_info():

    print(raw_data_train.info()) #to show datas info

    print('-'*30)

    print(raw_data_test.info())

    print('-'*30)

print_info()

raw_data_train.describe(include = 'all')
#data cleaning - Correcting, completing, creating, and converting

#correcting = correct wrong values and outlier



#completing = complete NaN datas

#fill all the nan datas

for data in raw_data:

    data['Age'].fillna(data['Age'].mean(), inplace = True) #fill age with mean

    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace = True) #fill embarked with the mode as no number,[0] is impt

    data['Fare'].fillna(data['Fare'].median(),inplace = True) #fill fare with median

#drop cabin feature as too many unknown from training dataset (test dataset remains same as no effect result)

raw_data_train.drop(['PassengerId','Cabin','Ticket','Name'],axis = 1,inplace=True) #if inplace=false means not replacing original data

print_info()
#Creating - create new features for analysis

for data in raw_data:

    data['FamilySize'] = data['Parch'] + data['SibSp'] + 1

    data['Alone'] = 1 #alone = 1, not alone = 0

    data['Alone'].loc[data['FamilySize'] > 1] = 0   

    data['FareBin'] = pd.qcut(data['Fare'], 5) #seperate fare into 5 classes

    data['AgeBin'] = pd.cut(data['Age'].astype(int),5) #seperate age into 5 classes

print_info()
#converting - to require data

#convert string data (categoricaEl) to int data through Label encoder

le = LabelEncoder()

for data in raw_data:

    data['SexCode'] = le.fit_transform(data['Sex'])

    data['EmbarkedCode'] = le.fit_transform(data['Embarked'])

    data['AgeBinCode'] = le.fit_transform(data['AgeBin'])

    data['FareBinCode'] = le.fit_transform(data['FareBin'])

print_info()
#Split training datas to train and validation set

data_bin = ['Pclass','SexCode','AgeBinCode','FamilySize','FareBinCode','EmbarkedCode','Alone']

prediction = ['Survived']

x_train,x_val,y_train,y_val = train_test_split(raw_data_train[data_bin],raw_data_train[prediction],

                                               test_size = 0.25,random_state = 42)

x_train.head()
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(raw_data_train)
#seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html

#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'EmbarkedCode', y = 'Survived', data=raw_data_train, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=raw_data_train, ax = saxis[0,1])

sns.barplot(x = 'Alone', y = 'Survived', order=[1,0], data=raw_data_train, ax = saxis[0,2])



sns.pointplot(x = 'FareBin', y = 'Survived',  data=raw_data_train, ax = saxis[1,0])

sns.barplot(x = 'AgeBinCode', y = 'Survived',  data=raw_data_train, ax = saxis[1,1])

sns.barplot(x = 'FamilySize', y = 'Survived', data=raw_data_train, ax = saxis[1,2])
XGBmodel = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(x_train, y_train)

XGBmodel.fit(x_train,y_train)

pred = XGBmodel.predict(x_val)
accuracy = accuracy_score(y_val, pred)

print(accuracy) #validation accuracy
#predict the test score

raw_data_test['Survived'] = XGBmodel.predict(raw_data_test[data_bin]) #create new column called survive on test.csv

submit = raw_data_test[['PassengerId','Survived']] #submit only passengerId and survived column

submit.to_csv("../working/submit.csv", index=False)