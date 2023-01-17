# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic_data.csv')
data.shape
data.dtypes
data.head(5)
data = data.drop(['Name','Ticket','Cabin'],axis=1)

missing_data = data.isnull() 
for column in missing_data.columns:

    print(missing_data[column].value_counts())
mean_ages = data.groupby(['Sex','Pclass'])['Age'].mean()

mean_ages

def remove_na_ages(row):

    if pd.isnull(row['Age']):

        return mean_ages[row['Sex'],row['Pclass']]

    else:

        return row['Age']

        

data['Age'] = data.apply(remove_na_ages,axis='columns')



    
data.dropna(inplace=True)

data.shape
data = pd.get_dummies(data,columns=['Embarked'])

data.rename(columns={"Embarked_C":"Cherbourg","Embarked_Q":"Queenstown","Embarked_S":"SouthHampton"})
data['Age'] = data['Age'].astype(int)

maximum_age = data['Age'].max()

age_labels = ['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79']

data['Age'] = pd.cut(data.Age,range(0,maximum_age+1,10),right=True,labels=age_labels)
survived_passengers = data[data['Survived']==1]

print("The total number of survived passengers are " + str(len(survived_passengers)))
data.groupby(['Pclass','Sex']).size().unstack()
passengers_survived_by_class_and_sex = survived_passengers.groupby(['Pclass','Sex'])['Survived'].sum().unstack()

print(passengers_survived_by_class_and_sex)

passengers_survived_by_class_and_sex.plot.bar(use_index=True,title='Survived passengers in each class by gender',grid=True)

mean_survivors_by_class = data.groupby('Pclass').Survived.mean()

print(mean_survivors_by_class)

mean_survivors_by_class.plot.bar()
mean_survivors_by_gender = data.groupby('Sex').Survived.mean()

print(mean_survivors_by_gender)

mean_survivors_by_gender.plot(kind='bar')
survived_passengers_by_age = data.groupby('Age')['Survived'].sum()

print(survived_passengers_by_age)

survived_passengers_by_age.plot.bar()
mean_survived_passengers_by_age = data.groupby(['Age'])['Survived'].mean()

print(mean_survived_passengers_by_age)

mean_survived_passengers_by_age.plot.bar()
survived_passengers_by_age = data.groupby(['Age','Sex'])['Survived'].mean().unstack()

print(survived_passengers_by_age)

survived_passengers_by_age.plot.bar()