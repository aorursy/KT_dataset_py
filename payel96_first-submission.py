# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

# Reading the data

train_data = pd.read_csv('../input/train.csv')

train_data.head()
train_data.describe()
#dropping off columns - Cabin and Ticket

train_data = train_data.drop('Name', axis=1)

train_data.head()

#since many "Age" column have missing values, we need to fill them 

train_data.Age = train_data.Age.fillna(train_data.Age.mean())

train_data.describe()



#you can see the count of Age has increased to 891 from 714

#some analysis - Age , Passenger, Gender, Family , Fare, Embarkment

#Fare Analysis 

fare_df = train_data[['Survived', 'Pclass','Fare']]

fare_df.head()

#in the above output , we can see that Passenger class 3 has the least fare than 1 or 2 import
import matplotlib.pyplot as plt

%matplotlib inline 

import seaborn as sns



fig,(axis1,axis2) = plt.subplots(1,2,figsize=(20,10))

axis1.set_title('Fare VS Pclass')   #Pclass is of 3 types - 1, 2 ,3

axis2.set_title('Fare VS Survived') #survived is of 2 types - 0 , 1



sns.barplot(x='Pclass', y='Fare', data = fare_df, ax=axis1)

sns.barplot(x='Survived', y='Fare', data = fare_df, ax=axis2)
#Gender Analysis 

gender_df = train_data[['Survived', 'Pclass','Fare', 'Sex']]

gender_df.head()
#let's first convert the gender into binary values :- Male for 0 and female for 1



df = train_data.replace(to_replace='male',value=0 ).replace(to_replace='female',value=1)

df

fig,axis1 = plt.subplots(1,figsize=(10,10))

axis1.set_title('Survived VS Sex')   





sns.barplot(x='Sex', y='Survived', data = gender_df, ax=axis1)
#Pclass Analysis

pclass_df = train_data[['Survived', 'Pclass','Fare', 'Sex']]

pclass_df.head()

fig,axis1 = plt.subplots(1,figsize=(10,10))

axis1.set_title('Survived VS Pclass')   





sns.barplot(x='Pclass', y='Survived', data = pclass_df, ax=axis1)
#Family status 

#parch : no. of parents / children aboard the Titanic

parch_df = train_data[['Survived', 'Pclass','Fare', 'Sex', 'Parch']]

parch_df.head()

parch_df.tail()

#since Parch and SibSp has many different values so, we can't plot them on the graph directly in X-axis 



#We need to do Feature Engineering here,to create a new feature called "With_Family" 

#which would contain information on both chilren and spouses.



df['With_Family'] = (df['Parch'].astype(bool) | df['SibSp'].astype(bool)).astype(int)

df.head()
fig,axis1 = plt.subplots(1,figsize=(10,10))

axis1.set_title('Survived VS With_Family')   





sns.barplot(x='With_Family', y='Survived', data = df, ax=axis1)
#Age analysis 

age_df = train_data[['Survived', 'Pclass','Fare', 'Sex', 'Age']]

age_df.head()
fig,axis1 = plt.subplots(1,figsize=(10,10))

axis1.set_title('Survived VS Age')   





sns.distplot(age_df.Age, label='Survived', hist=True, kde=False)

#Embarkment analysis 

embarked_df = train_data[['Survived', 'Pclass','Fare', 'Sex', 'Age', 'Embarked']]

embarked_df.head()  # Embarkement is of 3 types - C = Cherbourg, Q = Queenstown, S = Southampton
fig,axis1 = plt.subplots(1,figsize=(10,10))

axis1.set_title('Survived VS Embarked')   





sns.barplot(x='Embarked', y='Survived', data = embarked_df, ax=axis1)

train_data.Embarked.value_counts()
sns.heatmap(df.corr(), linecolor='white', annot=True) 
#The negative correlation coefficient in the matrix implies that if one of the values increases the other decreases.