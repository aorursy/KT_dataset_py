# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from pandas.tools.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')
#gender_submission = pd.read_csv('../input/gender_submission.csv')
#Information about data.  This steps help us find data values
train.info()



#Finding the data head
train.head()

print('Cabin')
train["Cabin"].value_counts()
print('Ticket')
train["Ticket"].value_counts()


#Finding the Value For Objects 
print('Sex')
train["Sex"].value_counts()



#Finding the Value For Objects 
print('Embarked Values')
train["Embarked"].value_counts()




train.describe()
#Ploting Histogram 

import matplotlib.pyplot as plt

train.hist(bins=50, figsize=(20,15))
plt.show()
#Finding corelation between Survived
corr_matrix = train.corr()
print(corr_matrix["Survived"].sort_values(ascending=False))
#SibSp /Age / Pclass have negative cor
siblins_plot = train.pivot_table(index="SibSp",values="Survived")
siblins_plot.plot.bar()
plt.show()
class_pivot = train.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()
#the proportion of people in first class that survived is much higher
#TO DO Include the number of people on each group that survived
sex_pivot = train.pivot_table(index = "Sex", values = "Survived")
sex_pivot.plot.bar()
plt.show()
#the pivot table aggregates groups and applies a function to those
#The proportion of women that survived is 70% compared to aprox 20% for males
#TO DO: Include in the bar the number o people in each group
survived = train[train["Survived"]==1]
died = train[train["Survived"]==0]
survived["Age"].plot.hist(alpha = 0.5,color = 'green',bins = 50)
died["Age"].plot.hist(alpha = 0.5,color = 'black',bins = 50)
plt.legend(['Survived','Died'])
plt.show()
#Shows the distributions of of population by the age and if they survived and Died
#Bucketing the population into segments
#This allows to make a story 
def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)
train.head()
age_category_plot = train.pivot_table(index="Age_categories",values='Survived')
age_category_plot.plot.bar()
plt.show()
#Infants had the highest survival rate
#Hot Encoding for Pclass , Sex,Age_Categoies 
def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis = 1)
    return df

for column in ["Pclass","Sex","Age_categories"]:
    train = create_dummies(train,column)
    test = create_dummies(test,column)    
    
train.head()
#Preparing the dataframe to train it
list(train)
#Getting only engineered columns
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split

model_columns = [ 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Sex_female',
 'Sex_male',
 'Age_categories_Missing',
 'Age_categories_Infant',
 'Age_categories_Child',
 'Age_categories_Teenager',
 'Age_categories_Young Adult',
 'Age_categories_Adult',
 'Age_categories_Senior',
 'Pclass_1',
 'Pclass_2',
 'Pclass_3',
 'Sex_female',
 'Sex_male',
 'Age_categories_Missing',
 'Age_categories_Infant',
 'Age_categories_Child',
 'Age_categories_Teenager',
 'Age_categories_Young Adult',
 'Age_categories_Adult',
 'Age_categories_Senior']

lr = LogisticRegression()
lr.fit(train[model_columns],train["Survived"])
#Dividing now the train datafrain between train and test set (Since the given test df doesn't have the target column)
all_X = train[model_columns]
all_y = train['Survived']
train_X, test_X, train_y, test_y = train_test_split(all_X,all_y, test_size = 0.20, random_state = 0)
test_X.describe()
#training the data in the training set and predicting in the test set
lr = LogisticRegression()
lr.fit(train_X,train_y)
predictions = lr.predict(test_X)
#measuring accuracy of predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y,predictions)
print(accuracy)
lr.fit(all_X,all_y)
holdout_predictions = lr.predict(test[model_columns])
submission_df = {"PassengerId": test["PassengerId"],
                              "Survived":holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.head()