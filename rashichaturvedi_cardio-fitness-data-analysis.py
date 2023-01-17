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
df= pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
#Check first five rows of the data
df.head()
# Describe the data to know its summary
df.describe()
#Check the information of the data
df.info()
#Check the shape of the data
df.shape
#Check for null values
df.isna().any()
# Data Visualisation
import matplotlib.pyplot as plt
df.hist(figsize=(20,30))
import seaborn as sns
sns.boxplot(x="Gender", y="Age", data=df)
pd.crosstab(df['Product'],df['Gender'] )
pd.crosstab(df['Product'],df['MaritalStatus'] )
sns.countplot(x="Product", hue="Gender", data=df)
pd.pivot_table(df, index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'], aggfunc=len)
pd.pivot_table(df,'Income', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])
pd.pivot_table(df,'Miles', index=['Product', 'Gender'],
                     columns=[ 'MaritalStatus'])
sns.pairplot(df)
df['Age'].std()
df['Age'].mean()
sns.distplot(df['Age'])
df.hist(by='Gender',column = 'Age')
df.hist(by='Gender',column = 'Income')
df.hist(by='Gender',column = 'Miles')
df.hist(by='Product',column = 'Miles', figsize=(20,30))
corr = df.corr()
corr
sns.heatmap(corr, annot=True)
# Simple Linear Regression


#Load function from sklearn
from sklearn import linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

y = df['Miles']
x = df[['Usage','Fitness']]

# Train the model using the training sets
regr.fit(x,y)

regr.coef_
regr.intercept_
# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness