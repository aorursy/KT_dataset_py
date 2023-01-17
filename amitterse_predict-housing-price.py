#invite people for the Kaggle party

import pandas as pd

from pandas.plotting import scatter_matrix



import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

df_train = pd.read_csv('../input/train.csv')

#check the decoration

df_train.columns

df_train.head(10)

df_train.describe(include='all')

df_train.describe(include=[np.number])

df_train.describe(include=[np.object])

#descriptive statistics summary

df_train['SalePrice'].describe()


#histogram

sns.distplot(df_train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew())

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

#data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

#plt.xticks(rotation=90);
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data: xi-xmean/standar deviation

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation will help to form a bell shape curve

df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

#fig = plt.figure()

#res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ##Accuracy score is for classification

from sklearn.metrics import explained_variance_score ## This is for linear regression



df_test=pd.read_csv('../input/test.csv')



#Create X_test

X_test=df_test[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]

X_test.head()



#Checking missing value in test data set

total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)

total_missing_values_X_test



#Checking the missing Garage Cars record

X_test[X_test['GarageCars'].isnull()]



#Updating the missing value to mean value

X_test.at[660,'TotalBsmtSF'] = 1046.12



#Verifying the missing value in TotalBsmtSF

X_test[X_test['TotalBsmtSF'].isnull()]





X_test.at[1116,'GarageCars'] = 2



#Checking missing value in test data set again

total_missing_values_X_test=X_test.isnull().sum().sort_values(ascending=False)

total_missing_values_X_test



X_train=df_train[['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']]

Y_train=df_train[['SalePrice']]

                  

#Checking missing value in test data set again

total_missing_values_X_train=X_train.isnull().sum().sort_values(ascending=False)

total_missing_values_X_train



regressor=LinearRegression()

regressor.fit(X_train,Y_train)

features=['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']



Y_pred=regressor.predict(X_train)

Y_pred_test=regressor.predict(X_test[features])

Y_pred_df=pd.DataFrame(Y_pred) 

Y_pred_test_df=pd.DataFrame(Y_pred_test) 

Y_pred_df



explained_variance_score(Y_train,Y_pred_df)

#Y_pred_test_df=pd.DataFrame({'SalesPrice':Y_pred_test[0:]})

Y_pred_test_df.to_csv('sample_submission_pred.csv', index=False)

#Y_pred_test_df=pd.DataFrame({'Id':df_test['Id'],'SalesPrice':Y_pred_test})

#X_test['OverallQual','YearBuilt','GrLivArea','TotalBsmtSF','GarageCars']

#Y_pred_test_df=pd.DataFrame({'Id':df_test['Id'],'Sales':Y_pred_test})























## Now with test data predicting values 