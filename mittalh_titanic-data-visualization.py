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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv("../input/titanic/train.csv")   # reading CSV file

df.head(4)      
df.info()   # analyzing dtypes and null values 
plt.subplots_adjust(left=14,right=15) 

plt.subplot(2,2,1)

sns.countplot(df['Survived'])

plt.subplot(2,2,2)

sns.countplot(df['Survived'],hue=df['Sex'])

plt.subplot(2,2,3)

sns.countplot(df['Survived'],hue=df['Pclass'])

plt.subplot(2,2,4)

sns.countplot(df['Survived'],hue=df['Embarked'])
plt.subplots_adjust(left=24,right=25) 

plt.subplot(2,2,1)

sns.countplot(df['Sex'])

plt.subplot(2,2,2)

sns.countplot(df['Pclass'],hue=df['Sex'])

plt.subplot(2,2,3)

sns.countplot(df['Sex'],hue=df['Embarked'])

plt.subplots_adjust(left=13,right=15) 

plt.subplot(1,2,1)

plt.hist(df['Age'],bins=15)

plt.xlabel("Age")

plt.ylabel("Frequency")

plt.subplot(1,2,2)

plt.hist(df['Fare'],bins=15)

plt.xlabel("Fare")

plt.ylabel("Frequency")
plt.subplots_adjust(left=13,right=15) 

plt.subplot(1,2,1)

sns.boxplot(df['Sex'],df['Age'])

plt.subplot(1,2,2)

sns.boxplot(df['Survived'],df['Age'],hue=df['Sex'])
plt.subplots_adjust(left=13,right=15) 

plt.subplot(1,2,1)

sns.boxplot(df['Embarked'],df['Age'])

plt.subplot(1,2,2)

sns.boxplot(df['Embarked'],df['Age'],hue=df['Sex'])
plt.subplots_adjust(left=13,right=15) 

plt.subplot(1,2,1)

sns.boxplot(df['Pclass'],df['Age'])

plt.subplot(1,2,2)

sns.boxplot(df['Pclass'],df['Age'],hue=df['Sex'])
plt.subplots_adjust(left=8,right=9) 

plt.subplot(1,2,1)

sns.boxplot(df['Pclass'],df['Fare'])

plt.subplot(1,2,2)

sns.boxplot(df['Pclass'],df['Fare'],hue=df['Sex'])
plt.subplots_adjust(left=13,right=14) 

plt.subplot(1,2,1)

sns.boxplot(df['Sex'],df['Fare'])

plt.subplot(1,2,2)

sns.boxplot(df['Survived'],df['Fare'],hue=df['Sex'])