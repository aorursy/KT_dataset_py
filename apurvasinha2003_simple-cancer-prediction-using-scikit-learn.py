# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
from sklearn.linear_model import LogisticRegression as lm
from sklearn.model_selection import train_test_split
from sklearn import tree
from matplotlib import pyplot as plt    
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
cancer = pd.read_csv('../input/data.csv')
print(cancer.head(4))
cancer.shape
#dropping and removing unwanted columns
cancer = cancer.drop(['id','Unnamed: 32'],axis=1)
#convert diagnosis into binary(1=M, 0=B)
diagnosis ={'diagnosis':{'M':1,'B':0}}
cancer =cancer.replace(diagnosis)

print(cancer.head(1))
#plot to check for noral distribution of data
plt.figure()
cancer.diff().hist(alpha=0.5,figsize=(14, 14))
plt.legend()
plt.show()
#splitting the data
train,test=train_test_split(cancer,test_size=0.3,random_state=14)
print(type(train))
#Function which takes all features into consideration
def building_model_all(model,train,test):
    y_train=train[['diagnosis']]
    x_train=train[train.columns[1:]]
    y_test=test[['diagnosis']]
    x_test=test[test.columns[1:]]
    model.fit(x_train,y_train)
    y_prd = model.predict(x_test)
    #print("coefficient: "+ str(model.coef_))
    print("\nmean Squared error : \n")
    print(mean_squared_error(y_test,y_prd))
    print('\nVariance score :\n')
    print(r2_score(y_test,y_prd))
    print("\naccuracy score: \n")
    print(accuracy_score(y_test,y_prd))


#selective features for model
def building_model_selective(model,train,test):
    y_train = train[['diagnosis']]
    names = ['radius_mean','perimeter_mean', 'area_mean', 'compactness_mean']
    x_train = train[names]
    y_test = test[['diagnosis']]
    x_test = test[names]
    model.fit(x_train,y_train)
    y_prd = model.predict(x_test)
    #print("\ncoefficient: "+ str(model.coef_))
    print("\nmean Squared error : \n")
    print(mean_squared_error(y_test,y_prd))
    print('\nVariance score :\n')
    print(r2_score(y_test,y_prd))
    print("\naccuracy score: \n")
    print(accuracy_score(y_test,y_prd))
#logistic regression
model = lm()
building_model_all(model,train,test)
building_model_selective(model,train,test)
#decision tree
model_tree = tree.DecisionTreeClassifier()
building_model_all(model_tree,train,test)
building_model_selective(model_tree,train,test)