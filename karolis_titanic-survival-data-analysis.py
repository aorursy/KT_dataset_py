# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv as csv



# Input data files are available in the "../input/" directory.



df = pd.read_csv('../input/train.csv')



print(df[df.Age > 35].head(20))
df['SexGroup'] = df['Sex'].map({'male': 0, 'female': 1})



total_male = df[df['Sex'] == 'male'].PassengerId.count()

total_female = df[df['Sex'] == 'female'].PassengerId.count()



survived_male = df[(df['Sex'] == 'male') & (df['Survived'] == True)].PassengerId.count()

survived_female = df[(df['Sex'] == 'female') & (df['Survived'] == True)].PassengerId.count()



survived_ratio_male = survived_male / total_male

survived_ratio_female = survived_female / total_female



print(survived_ratio_male, survived_ratio_female)
### Group passengers by Pclass and inside there, Fare
# Check passengers survival by Age

df[df.Age.notnull()].Age.describe()



# Based on description data, we can group passengers in groups of 20 years

# i.e. <20, 20< age <40, >40