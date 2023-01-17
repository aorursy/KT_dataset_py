# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

color=sns.color_palette()

sns.set_style('darkgrid')

%matplotlib inline

import matplotlib.pyplot as plt

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn =ignore_warn #warnings from sklearn and seaborn
test=pd.read_csv('../input/test_technidus.csv')

df_train=pd.read_csv('../input/train_technidus.csv')
df_train.head(5)
df_train.shape
df_train.info()
test.info()
#Print out all the columns that have not more than 30% null values

nn_cols=[col for col in df_train.columns if df_train[col].count()>=0.7*len(df_train)]

print(nn_cols)
train=df_train[nn_cols]

test=test[nn_cols]
train.isnull().sum()
test.isnull().sum()
train.nunique()
test.nunique()
train.describe()
test.describe()
cat_col=['CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren', 'BikeBuyer']

num_col=['BirthYear', 'BirthMonth', 'YearlyIncome', 'AveMonthSpend']
#Convert BirthDate to Year,Month

train['BirthYear']=pd.to_datetime(train['BirthDate']).dt.year;

train['BirthMonth']=pd.to_datetime(train['BirthDate']).dt.month;

train.drop(['BirthDate'],axis=1,inplace=True)
test['BirthYear']=pd.to_datetime(test['BirthDate']).dt.year;

test['BirthMonth']=pd.to_datetime(test['BirthDate']).dt.month;
Xtest=test.copy()
#Distribution of customers for each categorical variable

for col in cat_col:

    print(train[col].value_counts())

    print('')
for col in cat_col:

        fig = plt.figure(figsize=(6,6))

        ax = fig.gca()   

        counts = train[col].value_counts()

        counts.plot.bar(ax = ax, color = 'blue') 

        ax.set_title('Number of customers by ' + col)

        ax.set_xlabel(col) 

        ax.set_ylabel('Number of customers')

        plt.show()
sns.pairplot(train[num_col],diag_kind='kde')
plt.figure(figsize=(10,7))

sns.heatmap(train[num_col].corr(),annot=True)
#Checking the effect of each categorical varaible on the target

def plot_box(data, cols, col_y = None):

    for col in cols:

        plt.figure(figsize=(15,5))

        sns.boxplot(y=col_y, x=col, data=data)

        plt.ylabel(col_y) # Set text for the x axis

        plt.xlabel(col)# Set text for y axis

        plt.show()

        

plot_box(data=train,cols=cat_col,col_y='AveMonthSpend')
#Checking if BirthMonth can be a categorical variable

plot_box(data=train,cols=['BirthMonth'],col_y='AveMonthSpend')
#Heat map

corrmat= df_train.corr()

f, ax =plt.subplots(figsize=(5,4))

sns.heatmap(corrmat, square=True, annot=True)
del df_train['Suffix']
df_train.columns
test.columns
cols= ['CustomerID', 'CountryRegionName', 'BirthDate','Education', 'Occupation', 'Gender', 'MaritalStatus', 'HomeOwnerFlag',

       'NumberCarsOwned', 'NumberChildrenAtHome', 'TotalChildren','YearlyIncome', 'AveMonthSpend', 'BikeBuyer']

dataset=[df_train, test]

for data in dataset:

    train1= df_train[cols]

    test=test[cols]

train1.head(5)

train1.shape
test.shape
dataset=[train1, test]

for data in dataset:

    data["BirthDate"]=pd.to_datetime(data['BirthDate'], infer_datetime_format=True)

    data['year']=data['BirthDate'].dt.year

    data['Age']=1998-data['year']

train1.head(5).T
del train1['BirthDate']

del test['BirthDate']
sns.boxplot(data=train1, x='Occupation', y='AveMonthSpend')
sns.scatterplot(data=train1, x='Age', y='AveMonthSpend', hue='Gender')
sns.barplot(data=train1, x='MaritalStatus', y='AveMonthSpend')
column=['YearlyIncome','AveMonthSpend', 'Age']

def distplot(df, column, bins = 10, hist = False):

    for col in column:

        sns.distplot(df[col], bins=bins, rug=True, hist=hist)

        plt.title('Distribution for' + col)

        plt.xlabel(col)

        plt.ylabel('Frequency')

        plt.show()



distplot(train1, column, hist= True)
for data in dataset:

    data['Age']=np.log(data['Age'])

    data['YearlyIncome']=(data['YearlyIncome'])**0.5

    

column=['YearlyIncome','AveMonthSpend', 'Age']



distplot(train1, column, hist=True)
col=['CountryRegionName', 'Education', 'Occupation', 'Gender', 'MaritalStatus']

train=pd.get_dummies(train1, prefix=col, columns=col)

test=pd.get_dummies(test, prefix=col, columns=col)
del test['year']

del train['year']
train.shape
test.shape
test.columns
train.columns
cols=['HomeOwnerFlag', 'NumberCarsOwned',

       'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome', 'BikeBuyer',

       'Age', 'CountryRegionName_Australia',

       'CountryRegionName_Canada', 'CountryRegionName_France',

       'CountryRegionName_Germany', 'CountryRegionName_United Kingdom',

       'CountryRegionName_United States', 'Education_Bachelors ',

       'Education_Graduate Degree', 'Education_High School',

       'Education_Partial College', 'Education_Partial High School',

       'Occupation_Clerical', 'Occupation_Management', 'Occupation_Manual',

       'Occupation_Professional', 'Occupation_Skilled Manual', 'Gender_F',

       'Gender_M', 'MaritalStatus_M', 'MaritalStatus_S']

test1=test[cols]
test1.shape
f,ax=plt.subplots(figsize=(18,15))

sns.heatmap(train.corr(),linewidth=2.0, ax=ax, annot=True)

ax.set_title('Correlation Matrix')
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from math import sqrt
feature_col=['HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome',

       'TotalChildren', 'YearlyIncome','BikeBuyer', 'Age',

       'CountryRegionName_Australia', 'CountryRegionName_Canada',

       'CountryRegionName_France', 'CountryRegionName_Germany',

       'CountryRegionName_United Kingdom', 'CountryRegionName_United States',

       'Education_Bachelors ', 'Education_Graduate Degree',

       'Education_High School', 'Education_Partial College',

       'Education_Partial High School', 'Occupation_Clerical',

       'Occupation_Management', 'Occupation_Manual', 'Occupation_Professional',

       'Occupation_Skilled Manual', 'Gender_F', 'Gender_M', 'MaritalStatus_M',

       'MaritalStatus_S']

predicted_class_names=['AveMonthSpend']

X=train[feature_col].values

y=train[predicted_class_names].values 

split_test_size=0.30

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=split_test_size, random_state=42)
train.shape
print("{0:0.2f}% in training set".format((len(X_train)/len(train.index)) * 100))

print("{0:0.2f}% in test set".format((len(X_test)/len(train.index)) * 100))
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train=ss.fit_transform(X_train)

X_test=ss.transform(X_test)

test1=ss.transform(test1)
reg=LinearRegression()

reg.fit(X_train, y_train)
regpred=reg.predict(X_train)

regpred_test=reg.predict(X_test)

print("Accuracy on training set:{:.7f}".format(reg.score(X_train, y_train)))

print("Accuracy on test set: {:.7f}".format(reg.score(X_test, y_test)))



MSE= mean_squared_error(y_train, regpred)

MSE_test= mean_squared_error(y_test, regpred_test)

print("MSE:", MSE)

print("MSE Test:", MSE_test)

RMSE=sqrt(MSE)

RMSE_test=sqrt(MSE_test)

print("RMSE:", RMSE)

print("RMSE Test:", RMSE_test)
gbr=GradientBoostingRegressor (max_depth=5, loss='huber', n_estimators=1000, learning_rate=0.01)

gbr.fit(X_train, y_train.ravel())
print("Accuracy on training set: {:.7f}".format(gbr.score(X_train, y_train.ravel())))

print("Accuracy on test set: {:.7f}".format(gbr.score(X_test, y_test)))

gbrpred=gbr.predict(X_train)

gbrpred_test=gbr.predict(X_test)



MSE= mean_squared_error(y_train, gbrpred)

MSE_test= mean_squared_error(y_test, gbrpred_test)

print("MSE:", MSE)

print("MSE Test:", MSE_test)

RMSE=sqrt(MSE)

RMSE_test=sqrt(MSE_test)

print("RMSE:", RMSE)

print("RMSE Test:", RMSE_test)
solution=gbr.predict(test1)

my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'AveMonthSpend': solution})

my_submission.to_csv('GradientBoostingMicrosoft.csv', index=False)
import xgboost as xgb

xgb=xgb.XGBRegressor(max_depth=5, n_estimators=100)

xgb.fit(X_train, y_train)
print("Accuracy on training set: {:.7f}".format(xgb.score(X_train, y_train.ravel())))

print("Accuracy on test set: {:.7f}".format(xgb.score(X_test, y_test)))

xgbpred=xgb.predict(X_train)

xgbpred_test=xgb.predict(X_test)



MSE= mean_squared_error(y_train, xgbpred)

MSE_test= mean_squared_error(y_test, xgbpred_test)

print("MSE:", MSE)

print("MSE Test:", MSE_test)

RMSE=sqrt(MSE)

RMSE_test=sqrt(MSE_test)

print("RMSE:", RMSE)

print("RMSE Test:", RMSE_test)
solution=xgb.predict(test1)

my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'AveMonthSpend': solution})

my_submission.to_csv('XgboostMicrosoft.csv', index=False)
import xgboost as xgb

xgb=xgb.XGBRegressor(max_depth=5, n_estimators=1000, learning_rate=0.01, reg_alpha=0.5, reg_lambda=0.9)

xgb.fit(X_train, y_train)
print("Accuracy on training set: {:.7f}".format(xgb.score(X_train, y_train.ravel())))

print("Accuracy on test set: {:.7f}".format(xgb.score(X_test, y_test)))

xgbpred=xgb.predict(X_train)

xgbpred_test=xgb.predict(X_test)



MSE= mean_squared_error(y_train, xgbpred)

MSE_test= mean_squared_error(y_test, xgbpred_test)

print("MSE:", MSE)

print("MSE Test:", MSE_test)

RMSE=sqrt(MSE)

RMSE_test=sqrt(MSE_test)

print("RMSE:", RMSE)

print("RMSE Test:", RMSE_test)
solution=xgb.predict(test1)

my_submission=pd.DataFrame({'CustomerID':test.CustomerID,'AveMonthSpend': solution})

my_submission.to_csv('XgboostMicrosoft02.csv', index=False)
test=Xtest