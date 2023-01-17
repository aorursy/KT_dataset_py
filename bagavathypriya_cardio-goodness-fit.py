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
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("../input/cardiogoodfitness/CardioGoodFitness.csv")
df.head()
df.describe()
df.info()
df.shape
sns.pairplot(df)
sns.heatmap(df.corr())
pd.crosstab(df['Product'],df['MaritalStatus'])
pd.crosstab(df['Product'],df['Gender'])
pd.pivot_table(df,'Income',index=['Product','Gender'],columns=['MaritalStatus'])
sns.distplot(df['Miles'])
x=df[['Usage','Fitness']]

y=df['Miles']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(xtrain,ytrain)
score=reg.score(xtest,ytest)
score*100
score=reg.score(xtrain,ytrain)
score*100
