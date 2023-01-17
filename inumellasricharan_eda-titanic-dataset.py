# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train_set = pd.read_csv('/kaggle/input/titanic/train.csv', sep = ',', header = 0)
train_set.head()
train_set.tail()
train_set.isnull()
type(train_set.isnull())
sns.heatmap(train_set.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')

sns.countplot(x = 'Survived', data = train_set)
sns.countplot(x = 'Survived', hue = 'Sex', data = train_set)

#hue is used to show the relationship with two categorical variables
sns.countplot(x = 'Survived', hue = 'Pclass', data = train_set)
sns.distplot(train_set['Age'].dropna())#with kernel density estimation set to true

#plot the distribution of the Age column
sns.distplot(train_set['Fare'], kde = True, bins = 40)
sns.countplot(x = 'SibSp', data = train_set)
#an alternative way

train_set['Fare'].hist(color = 'blue', bins = 40, figsize = (8, 4))
plt.figure(figsize = (12, 12))

sns.boxplot(y = 'Age', x = 'Pclass', data = train_set)
#its time to replace those null values in the age column

def impute_age(row):

    

    age = row[0]

    pclass = row[1]

    

    if(pd.isnull(age)):

        

        if(pclass == 1):

            

            return 37

            

        elif(pclass == 2):

            

            return 28

            

        elif(pclass == 3):

    

            return 24

    

    else:

        

        return age

    

        #all those values are eyeballed by looking at the boxplot

        

train_set['Age'] = train_set[['Age', 'Pclass']].apply(impute_age, axis = 1)
sns.heatmap(train_set.isnull(), yticklabels = False)

#look at the heatmap again to see if we have replaced the age correctly
train_set.head()
train_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace = True, axis = 1)
train_set.head()
sex = pd.get_dummies(train_set['Sex'], drop_first = False)#one hot encoding!

sex.head()
embarked = pd.get_dummies(train_set['Embarked'], drop_first = False)

embarked.head()
train_set.drop(['Sex', 'Embarked'], inplace = True, axis = 1)

train_set = pd.concat([train_set, sex, embarked], axis = 1)



train_set.head()