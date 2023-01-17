# DATA ENGINEERING OF TITANIC RAW DATA (TEST DATA)
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
## Importing test Data

test=pd.read_csv('test_titanic.csv')

test.head()
test.shape
# We will start by checking missing values in each of the columns in our test dataset

# Missing Data

# We can do this by making use of or creating a simple heatmap to see where we have missing values in the test dataset

sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')
sns.distplot(test['Age'].dropna(),kde=False,color='darkred',bins=30)
test['Age'].hist(bins=30,color='darkred',alpha = 0.7)
sns.countplot(x='SibSp',data = test)
test['Fare'].hist(color = 'green', bins = 40, figsize=(8,4))
plt.figure(figsize=(12,7))

sns.boxplot(x='Pclass', y='Age',data = test, palette= 'winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]



    if pd.isnull(Age):



      if Pclass == 1:

          return 42



      elif Pclass == 2:

          return 26



      else:

          return 23   



    else:

       return Age
## Once we finish with the above code, we will now make use of the code below to fix those missing ages above

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
## To do this, we are going to recall our heatmap code and check if there are still missing age data or not

sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')
## Just for us to justify what we are have in our heatmap above 

## and to see where else we re still having missing data (e.g Embarked)

test.isnull().sum()
## Now, let's try and fill those only one missing data in Fare column first using "Mode"

test['Fare'] = test['Fare'].fillna(test['Fare'].mode()[0])
## Next, is the Cabin column but based on the fact that over 50% of the Cabin data are missing in train dataset above

## Hence, the most reasonable approach is to remove the entire Cabin Column in order to have the same result like

## train dataset

## Thus, to do this we have the following code

test.drop(['Cabin'],axis=1,inplace=True)
## Hence, to confirm that there are no missing values in any of the columns in our test dataset

## We try and run our heatmap code again

sns.heatmap(test.isnull(),yticklabels= False,cbar=False,cmap='viridis')
## Also again to justify what we have in our heatmap above 

## and to see that there are no missing values again in our train dataset

test.isnull().sum()
test.head()
test.shape
## we will now save the cleaned test dataset as "formulatedtest.csv" as follows:

test.to_csv('formulatedtest.csv',index=False)