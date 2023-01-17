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
# Load the Cardio Dataset



mydata = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
#Display the first five rows of the data

mydata.head()
#Five point summary of the data

mydata.describe().T
#Info about the data

mydata.info()
#Shape of the data

mydata.shape
#Checking for the null values 



mydata.isna().any()
import matplotlib.pyplot as plt

%matplotlib inline



mydata.hist(figsize=(20,30))
import seaborn as sns



sns.boxplot(x="Gender", y="Age", data=mydata)
pd.crosstab(mydata['Product'],mydata['Gender'] )
pd.crosstab(mydata['Product'],mydata['MaritalStatus'] )
sns.countplot(x="Product", hue="Gender", data=mydata)
pd.pivot_table(mydata, index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'], aggfunc=len)
pd.pivot_table(mydata,'Income', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'])
pd.pivot_table(mydata,'Miles', index=['Product', 'Gender'],

                     columns=[ 'MaritalStatus'])
sns.pairplot(mydata)
mydata['Age'].std()
mydata['Age'].mean()
sns.distplot(mydata['Age'])
mydata.hist(by='Gender',column = 'Age')
mydata.hist(by='Gender',column = 'Income')
mydata.hist(by='Gender',column = 'Miles')
mydata.hist(by='Product',column = 'Miles', figsize=(20,30))
corr = mydata.corr()

corr
sns.heatmap(corr, annot=True)
# Simple Linear Regression





#Load function from sklearn

from sklearn import linear_model



# Create linear regression object

regr = linear_model.LinearRegression()



y = mydata['Miles']

x = mydata[['Usage','Fitness']]



# Train the model using the training sets

regr.fit(x,y)
regr.coef_
regr.intercept_
# MilesPredicted = -56.74 + 20.21*Usage + 27.20*Fitness