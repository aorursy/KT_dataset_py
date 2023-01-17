import numpy as np
import pandas as pd
mydata = pd.read_csv('../input/cardio-fitness-data/CardioGoodFitness.csv')
mydata.head()
mydata.describe()
mydata.info()
import matplotlib.pyplot as plt
# %matplotlib inline

mydata.hist(figsize=(20,20))
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
mydata['Fitness'].value_counts()
mydata['Age'].mean()
sns.distplot(mydata['Age'])
mydata.hist(by='Gender',column = 'Age')
mydata.hist(by='Gender',column = 'Income')
mydata.hist(by='Gender',column = 'Miles')
mydata.hist(by='Product',column = 'Miles', figsize=(20,20))
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