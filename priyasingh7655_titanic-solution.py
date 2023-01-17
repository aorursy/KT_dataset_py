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
dataset= pd.read_csv('../input/titanic/train.csv')

dataset.head()
#Exploring the dataset

dataset.info
#EDA on dataset

dataset.isnull()
#Finding null values by plotting heatmap

import seaborn as sns



print(dataset.isna().sum())



sns.heatmap(dataset.isnull(),yticklabels = False,cmap= 'viridis')
#Dropping cabin column as it contains more than 77% of null values



dataset= dataset.drop(['Cabin'],axis=1)

dataset.head()
#Plotting some graphs to visualize the data better



sns.countplot(x='Survived',hue='Pclass',data=dataset)
sns.countplot(x='SibSp',data=dataset)
sns.countplot(x='Embarked',hue='Survived',data=dataset)
#Plotting boxplot on Age and Pclass feature to find the median of Age feature as it contains some null values

sns.boxplot(x='Pclass',y='Age',data=dataset)
#Handling the null values in Age feature by assign the average value to the null values

def handle_null(columns_val):

    age=columns_val[0]

    pclass=columns_val[1]

    if pd.isnull(age):

        if pclass==1:

            return 39

        if pclass==2:

            return 31

        else:

            return 27

    else:

        return age

    

#Function call

dataset["Age"] = dataset[['Age','Pclass']].apply(handle_null,axis=1)
#Dropping the null values from embarked column as it contains only 2 null values

dataset= dataset.dropna(axis=1)

dataset.head()