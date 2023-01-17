import numpy as np

import pylab as pl

import pandas as pd

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import cross_val_score, GridSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/health-insurance-cross-sell-prediction/train.csv")

test = pd.read_csv("../input/health-insurance-cross-sell-prediction/test.csv")
#Select feature column names and target variable we are going to use for training

Gender  = {'Male': 1,'Female': 0} 

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

train.Gender = [Gender[item] for item in train.Gender] 

print(train)
#Select feature column names and target variable we are going to use for training

Vehicle_Age  = {'> 2 Years': 0,'1-2 Year': 1,'< 1 Year': 2} 

  

# traversing through dataframe 

# Vehicle_Age column and writing 

# values where key matches 

train.Vehicle_Age = [Vehicle_Age[item] for item in train.Vehicle_Age] 

print(train)
#Select feature column names and target variable we are going to use for training

Vehicle_Damage  = {'Yes': 0,'No': 1} 

  

# traversing through dataframe 

# Vehicle_Age column and writing 

# values where key matches 

train.Vehicle_Damage = [Vehicle_Damage[item] for item in train.Vehicle_Damage] 

print(train)
train.info()

train[0:10]
#Select feature column names and target variable we are going to use for training

Gender  = {'Male': 1,'Female': 0} 

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

test.Gender = [Gender[item] for item in test.Gender] 

print(test)
#Select feature column names and target variable we are going to use for training

Vehicle_Damage  = {'Yes': 1,'No':0} 

  

# traversing through dataframe 

# Vehicle_Age column and writing 

# values where key matches 

test.Vehicle_Damage = [Vehicle_Damage[item] for item in test.Vehicle_Damage] 

print(test)
#Select feature column names and target variable we are going to use for training

Vehicle_Age  = {'> 2 Years': 0,'1-2 Year': 1,'< 1 Year': 2} 

  

# traversing through dataframe 

# Vehicle_Age column and writing 

# values where key matches 

test.Vehicle_Age = [Vehicle_Age[item] for item in test.Vehicle_Age] 

print(test)
test.info()

test[0:10]
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["Response"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
train = train[['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium',

'Policy_Sales_Channel','Vintage','Response']] #Subsetting the data

cor = train.corr() #Calculate the correlation of the above variables

sns.heatmap(cor, square = True) #Plot the correlation as heat map
#Select feature column names and target variable we are going to use for training

features=['Gender','Age','Driving_License','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Annual_Premium',

'Policy_Sales_Channel','Vintage']

target = 'Response'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(10).values
from sklearn.ensemble import RandomForestClassifier



# We define the RF model

rfcla = RandomForestClassifier(n_estimators=100,random_state=9,n_jobs=-1)



# We train model

rfcla.fit(train[features],train[target]) 



#Make predictions using the features from the test data set

predictions = rfcla .predict(test[features])



#Display our predictions

predictions
#Create a  DataFrame

submission = pd.DataFrame({'id':test['id'],'Response':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)