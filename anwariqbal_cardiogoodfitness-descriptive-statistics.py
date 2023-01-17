#Import NumPy and Pandas

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Load the Cardio Dataset into dataframe



df = pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')
#Display the first five rows of the data

df.head()
#Five point summary of the data

df.describe(include="all")
#Info about the data

df.info()
#Shape of the data

df.shape
#Checking for the null values 



df.isna().any()
import matplotlib.pyplot as plt

%matplotlib inline



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