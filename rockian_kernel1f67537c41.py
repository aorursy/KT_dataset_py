# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



#import scikit_learn as sklearn
#Loading data

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

gender_submission = pd.read_csv('../input/gender_submission.csv')
#Viewing top 4 records

train_data.head(4)

#gender_submission.describe()
#Quick summary of data

train_data.describe()
#data visualisation

#heatmap missing values

fig, ax = plt.subplots(figsize=(9,5))

sns.heatmap(train_data.isnull(), cbar=False, cmap="YlGnBu_r")

plt.show()

#filling the missing the missing age by 0

train_data.Age = train_data.Age.fillna(0)

train_data.head()
#Visualizing survival rate by embarkment using barplot

sns.barplot(x='Embarked', y='Survived', data=train_data)



#Visualizing survival rate by fare using catplot

%config InlineBackend.figure_format = "retina"

sns.catplot(x="Survived",y="Fare", data = train_data);



#Visualizing survival rate by age and sex using boxplot

sns.boxplot("Survived","Age", hue='Sex', data = train_data);
#Create categories of ages

ages = (0, 5, 12, 18, 25, 35, 70)

groups = ['Baby', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']

categories = pd.cut(train_data.Age, ages, labels=groups)

train_data.Age = categories

train_data.head(10)