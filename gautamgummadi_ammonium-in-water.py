# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("/kaggle/input/ammonium-prediction-in-river-water/train.csv")
test = pd.read_csv("/kaggle/input/ammonium-prediction-in-river-water/test.csv")
def explore(df):

    print("Head of the dataset: ",df.head())

    print("Shape of the dataset: ",df.shape)

    print("# Null values columnwise ",df.isna().sum())

    for i in list(df.columns):

        print("Range of Column values: ", min(df[i]),"-",max(df[i]))

        j = ((df[i].isna().sum())/147)*100

        print("Station ",i," has ",round(j,2),'% of null values')

        print("Mean of Station",i,":",df[i].mean())

        df[i]=df[i].fillna(df[i].mean())

        if(j > 10):

            print("The Station ",i,"has too many null values, needs imputation or delete the column")

            ## Imputing NULL values with mean for plotting purpose

            df[i].fillna(df[i].mean(),inplace=True)

        plt.subplots(1, 1, sharex='col')

        sns.distplot(df[i])

    corr = df.corr()

    sns.heatmap(corr)

    

        

    
explore(train)
explore(test)
## Seperating Target from train set

y = train['target']

train = train.drop('target',axis=1)

train = train.drop('Id',axis=1)

test = test.drop('Id',axis=1)
# splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(train, y, test_size=0.3, 

                                                    random_state=1) 
print("X_train shape: ", X_train.shape)

print("y_train shape: ", y_train.shape)

print("X_val shape: ", X_val.shape)

print("y_val shape: ", y_val.shape)

from sklearn import datasets, linear_model, metrics



model1 = linear_model.LinearRegression() 

model1.fit(X_train,y_train)
# regression coefficients 

print('Coefficients: \n', model1.coef_) 

  

pred_train = model1.predict(X_train)

pred_val = model1.predict(X_val)
## Accuracy

print('Train Accuracy: ', model1.score(X_train,y_train))

print('Test Accuracy: ', model1.score(X_val,y_val))
## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model1.predict(X_train), model1.predict(X_train) - y_train, 

            color = "green", s = 10, label = 'Train data') 

  

## plotting residual errors in test data 

plt.scatter(model1.predict(X_val), model1.predict(X_val) - y_val, 

            color = "blue", s = 10, label = 'Validation data') 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 5, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 
predictions = model1.predict(test)

plt.subplots(1,1)

sns.distplot(y_train,label='train')

sns.distplot(y_val,label='validation')

sns.distplot(predictions,label='predicted')

plt.legend()