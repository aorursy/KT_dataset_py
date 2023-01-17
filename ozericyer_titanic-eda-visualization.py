# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns; sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_train=pd.read_csv("/kaggle/input/titanic/train.csv")
data_gender=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
data_test=pd.read_csv("/kaggle/input/titanic/test.csv")
data_train
data_gender
data_test
pd.options.display.max_rows=None
pd.options.display.max_columns=None
display(data_train)
pd.reset_option("display.max_rows")
pd.reset_option("display.max_columns")
pd.get_option("display.max_rows")
pd.set_option("display.max_rows",20)
display(data_test)
data_train, data_gender, data_test
data_train.info()
#There are missing values in cabin and age.
data_test.info()
#There are missing values in age and cabin.
data_train.columns, data_gender.columns, data_test.columns
len(data_train.columns),len(data_gender.columns),len(data_test.columns)
display(data_train.tail())
display(data_gender.tail())
display(data_test.tail())
data_merged=pd.merge(data_gender,data_test,on="PassengerId")
data_merged
data_train.columns==data_merged.columns
data=pd.concat([data_train,data_merged],ignore_index=True)
data
data.info()
data.describe().T

data["Age"].mean()
data.Age.median()
data.Age.max()
perc = [.10,.20,.40,.60,.80]
include = ['float','integer']
desc = data.describe(percentiles = perc, include = include)
desc.T
c=data.corr()
c
c[c<1].abs().max() 
#Getting the max values of correlation coefficient (as absolute values) for each variable.
a=c.abs()<1
c.abs()[a].max()
# Getting the max values of correlation coefficient 
#(as absolute values) for each variable (Another method)
# We see relatively strong relation between fare and 
#passenger class.
a=c.abs()<1
b=c.abs()>.5
c.abs()[a&b]
# a method to see absolute values of correlation <1 and >0.5
data.isna().sum()
data[data.Cabin.isna()]
# Getting the nan values in cabin column
# We see that we have 1014 rows as nan values
data[data.Cabin.isna()==False]
# Getting the data excluding nan values
#1)for cabin:
data["Cabin"].value_counts()
#2)for name:
data["Name"].value_counts()
# We see two repeating names here.
data[(data.Name=="Connolly, Miss. Kate")|(data.Name=="Kelly, Mr. James")]
# We see that they have different information, so they are not the same persons.
#for tickets:
data.Ticket.value_counts()
#There are also  some repeating values.
data[data.Ticket=="CA. 2343"]
#We can say for CA. 2343 that the people who have the same ticket number are from the same family.
data.Survived.unique(), data.Sex.unique(), data.Pclass.unique(), data.SibSp.unique(), data.Parch.unique(), data.Embarked.unique()
data[(data.Parch==0)&(data.SibSp==0)]
#We see that 790 people travelled alone.
data[(data.Parch>=1)|(data.SibSp>=1)]["Survived"].mean()
data["Survived"].mean()
data["Survived"][(data.Parch==0)&(data.SibSp==0)].mean()
data.Survived[data.Sex=="female"].mean()
data["Survived"].mean()
data.Survived[data.Age<18].mean()
data.Survived.mean()
data.Survived[data.Pclass==1].mean()
data.Survived[data.Pclass==2].mean()
data.Survived[data.Pclass==3].mean()
data.columns
dat=data.copy()
dat.rename(columns={"SibSp":"kardes_es","Parch":"Eb_cocuk"},inplace=True)
dat
#First method of replace
dat.Survived.replace(0,"died",inplace=True)
dat.Survived.replace(1,"live",inplace=True)
dat
#Second method
dat.replace(["S","C","Q"],["Southampton","Cherbourg","Queenstown"],inplace=True)
dat
# data1.Kabin.fillna('Belirsiz', inplace = True)
# data1
data
plt.figure(figsize = (12,8))
sns.barplot(x='Embarked',y ='Fare',hue="Sex",data = data )
plt.xticks(rotation=30);

#For making much clear of the axis and table names
plt.xlabel('Embarked Names')
plt.ylabel('Amount of fare')
plt.title('Relationship of Embarked and Fare');
