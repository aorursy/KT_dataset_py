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
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")
train.info()

train[0:10]
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'].plot('bar', color='r',width=0.3,title='Survived', fontsize=10)

plt.xticks(rotation = 90)

plt.ylabel('Survived')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(10)

ax.yaxis.label.set_fontsize(10)

print(train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'][[1,2]])

print(train.groupby('Age').mean().sort_values(by='Survived', ascending=False)['Survived'][[4,5,6]])
#Frequency distribution of classes"

train_outcome = pd.crosstab(index=train["Survived"],  # Make a crosstab

                              columns="count")      # Name the count column



train_outcome
# for column

train['Age'] = train['Age'].replace(np.nan, 0)



# for whole dataframe

train = train.replace(np.nan, 0)



# inplace

train.replace(np.nan, 0, inplace=True)



print(train)
#Select feature column names and target variable we are going to use for training

Sex = {'male': 1,'female': 2} 

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

train.Sex = [Sex[item] for item in train.Sex] 

print(train)

test.info()

test[0:10]
print("Any missing sample in training set:",train.isnull().values.any())

print("Any missing sample in test set:",test.isnull().values.any(), "\n")
# for column

test['Age'] = train['Age'].replace(np.nan, 0)



# for whole dataframe

test = test.replace(np.nan, 0)



# inplace

test.replace(np.nan, 0, inplace=True)



print(test)
#Select feature column names and target variable we are going to use for training

Sex = {'male': 1,'female': 2} 

  

# traversing through dataframe 

# Gender column and writing 

# values where key matches 

test.Sex = [Sex[item] for item in test.Sex] 

print(test)


features=['Sex','Age']

target = 'Survived'
#This is input which our classifier will use as an input.

train[features].head(10)
#Display first 10 target variables

train[target].head(10).values
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000, random_state=42)



# We train model

mlp.fit(train[features],train[target]) 



# We train model

mlp.fit(train[features],train[target]) 

#Make predictions using the features from the test data set

predictions = mlp .predict(test[features])



#Display our predictions

predictions
# Test score

#score_svmcla = svmcla.score(test[features])

#print(score_svmcla)
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
#Convert DataFrame to a csv file that can be uploaded

#This is saved in the same directory as your notebook

filename = 'submission.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)