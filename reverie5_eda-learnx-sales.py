#Import lib

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns
#reading data

train= pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/train.csv')

test= pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/test_QkPvNLx.csv')

sample= pd.read_csv('/kaggle/input/womenintheloop-data-science-hackathon/sample_submission_pn2DrMq.csv')



print(train.shape, test.shape, sample.shape)
# check columns

print(train.columns)

print('-------------')

print(test.columns)
# take a look at the data

train.head()
# Descriptive Stats

train.Sales.describe()
train[['Sales']].boxplot()
#histogram

sns.distplot(train['Sales'])
# Positive skewness also mean, mean and median > mode

#mean, median, mode

print(train.Sales.mean())

print(train.Sales.median())

print(train.Sales.mode()[0])
#skewness and kurtosis

print("Skewness= ", train['Sales'].skew())

print("Kurtosis= ", train['Sales'].kurt())
#datatypes

train.dtypes
# ID vs Sales

#sns.regplot(x='ID', y='Sales', data=train)
train.ID.nunique()
train['ID'].hist()
# Day_No

train.Day_No.nunique()
train['Day_No'].hist()
sns.regplot(x='Day_No', y='Sales', data= train, ci=None)
# plotting on sample of dataset coz data is huge 

sampletrain= train.sample(1000)

sns.regplot(x='Day_No',y='Sales',data= sampletrain)
#3. Course_ID

train.Course_ID.nunique()
sns.regplot(x='Course_ID', y='Sales', data=sampletrain)
#4. Short_Promotion

train.Short_Promotion.nunique()
train.Short_Promotion.hist()
sns.boxplot(x='Short_Promotion',y='Sales',data=train)
sns.boxplot(x='Long_Promotion',y='Sales',data=train)
sns.boxplot(x='Course_Domain',y='Sales',data=train)
sns.boxplot(x='Course_Type',y='Sales',data=train)
sns.boxplot(x='Public_Holiday',y='Sales',data=train)
plt.subplots(figsize=(10,8))

sns.heatmap(train.corr(), annot= True)
#pairplot- high comput dont run



#sns.pairplot(train)
# train

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
# test

total_test = test.isnull().sum().sort_values(ascending=False)

percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data_test
sns.heatmap(train.isnull())
#standardizing data

from sklearn.preprocessing import StandardScaler

sales_scaled = StandardScaler().fit_transform(train['Sales'][:,np.newaxis])

low_range = sales_scaled[sales_scaled[:,0].argsort()][:10]

high_range= sales_scaled[sales_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#histogram and normal probability plot



from scipy.stats import norm

from scipy import stats



sns.distplot(train['Sales'], fit=norm)

fig = plt.figure()

res = stats.probplot(train['Sales'], plot=plt)
# to count the number of 0 in Sales

(train['Sales']==0).sum()
#applying log+1 transformation

train['Sales'] = np.log1p(train['Sales'])
#transformed histogram and normal probability plot

sns.distplot(train['Sales'], fit=norm)

fig = plt.figure()

res = stats.probplot(train['Sales'], plot=plt)