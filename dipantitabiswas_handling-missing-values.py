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
#importing necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
#load the data
train_df=pd.read_csv("../input/titanic/train.csv")
train_df.head()
#missing values
train_df.isnull().sum()
train_df.isnull().mean()
#filling the missing values with median
median=train_df['Age'].median()
train_df["Age_median"]=train_df['Age'].fillna(median)

sns.kdeplot(train_df["Age"])
sns.kdeplot(train_df["Age_median"])
#filling the missing value with random sample imputation
train_df["Age_random"]=train_df["Age"]
random_sample=train_df["Age"].dropna().sample(train_df["Age"].isnull().sum())
random_sample.index=train_df[train_df["Age"].isnull()].index
train_df.loc[train_df["Age"].isnull(),"Age_random"]=random_sample

sns.kdeplot(train_df["Age"])
sns.kdeplot(train_df["Age_random"])
#capturing NAN values with new feature
train_df["Age_nan"]=np.where(train_df["Age"].isnull,1,0)
#End of distribution imputer
a=train_df["Age"].mean()+3*train_df["Age"].std()
train_df["Age_end"]=train_df["Age"].fillna(a)
sns.kdeplot(train_df["Age"])
sns.kdeplot(train_df["Age_end"])
#arbitary value imputer
train_df["Age_ninety"]=train_df["Age"].fillna(90)
sns.kdeplot(train_df["Age"])
sns.kdeplot(train_df["Age_ninety"])
#Frequent Category Imputation
train_df['Embarked'].value_counts().plot.bar()
train_df["Embarked"].fillna(train_df['Embarked'].value_counts().index[0],inplace=True)
#Adding a variable capture NAN
train_df["Cabin_val"]=np.where(train_df["Cabin"].isnull(),1,0)
train_df["Cabin_cap"]=train_df["Cabin"].fillna(train_df['Cabin'].value_counts().index[0])
#Capturing NAN value with new Category
train_df["Cabin_new"]=np.where(train_df["Cabin"].isnull,"Missing",train_df["Cabin"])
train_df.head(20)