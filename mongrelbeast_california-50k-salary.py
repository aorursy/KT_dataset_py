# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

dataset = pd.read_csv("../input/adult.csv",na_values="?")

# Any results you write to the current directory are saved as output.
dataset.info()
dataset['income'].unique()
#convert '<=50k' to 0 and '>50k' to 1

def sal_cat(income):

    if income == '<=50K':

        return 0

    else:

        return 1

dataset['income']=dataset['income'].apply(sal_cat)
#divide data and label

x_data=dataset.drop('income',axis=1)

x_data.head()

y_label=dataset['income']
#count no. of null data 

x_data.isnull().sum()
#convert null data to string for encoding

x_data[x_data.workclass.isnull()]

x_data["workclass"].fillna("", inplace = True) 

x_data["occupation"].fillna("", inplace = True)

x_data["native.country"].fillna("", inplace = True)
#import tensorflow

import tensorflow as tf

#find categorical columns

cat_feat_mask=x_data.dtypes==object

cat_cols=x_data.columns[cat_feat_mask].tolist()

cat_cols
#encode categorical column using .cat.codes

x_data["workclass"]=x_data["workclass"].astype("category").cat.codes

x_data["occupation"]=x_data["occupation"].astype("category").cat.codes

x_data["native.country"]=x_data["native.country"].astype("category").cat.codes

x_data["education"]=x_data["education"].astype("category").cat.codes

x_data["marital.status"]=x_data["marital.status"].astype("category").cat.codes

x_data["relationship"]=x_data["relationship"].astype("category").cat.codes

x_data["race"]=x_data["race"].astype("category").cat.codes

x_data["sex"]=x_data["sex"].astype("category").cat.codes

#Note:missing data is now assigned with 0 after encoding

#apply mean on columns with missing data

from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=0, strategy='mean', axis=0)

x_data['workclass']=imp.fit_transform(x_data[['workclass']])

x_data['occupation']=imp.fit_transform(x_data[['occupation']])

x_data['native.country']=imp.fit_transform(x_data[['native.country']])
#training and testing data

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x_data,y_label,test_size=0.3)
#scale the data

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(xtrain)
xtrain=pd.DataFrame(data=scaler.transform(xtrain),

                   columns=xtrain.columns,

                   index=xtrain.index)

xtest=pd.DataFrame(data=scaler.transform(xtest),

                  columns=xtest.columns,

                  index=xtest.index)

#Convert raw data into feature column 

feat={}.fromkeys(['age','class','fnlwgt','edu','num','marital','occ','relat','race','sex','gain','loss','hours','count'], 0)

i=0

for item in feat:

    feat[item]=tf.feature_column.numeric_column(x_data.columns[i])

    i+=1
#input function

input_func=tf.estimator.inputs.pandas_input_fn(x=xtrain,y=ytrain,batch_size=10,num_epochs=1000,shuffle=True)
#feature columns

feat_cols=[feat['age'],feat['class'],feat['fnlwgt'],feat['edu'],feat['num'],feat['marital'],feat['occ'],feat['relat'],feat['race'],feat['sex'],feat['gain'],feat['loss'],feat['hours'],feat['count']]

feat_cols

#create model

model=tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes=2)
#train model

model.train(input_fn=input_func,steps=10000)
#input function  for prediction

pred_input_func=tf.estimator.inputs.pandas_input_fn(x=xtest,batch_size=10,num_epochs=1,shuffle=False)
#predict

pred_gen=model.predict(pred_input_func)
#list of prediction

predictions=list(pred_gen)
#grab only predicted value from list of prediction

final_preds=[pred['class_ids'][0] for pred in predictions]

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
print("Accuracy: %s%%" % (100*accuracy_score(ytest, final_preds)))

print(confusion_matrix(ytest,final_preds))

print(classification_report(ytest, final_preds))