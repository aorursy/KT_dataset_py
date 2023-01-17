#importing required libraries 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pylab 

import scipy.stats as stats
#reading data from csv files

test = pd.read_csv('../input/test-a102csv/Test_A102 (1).csv')

train = pd.read_csv('../input/train-a102csv/Train_A102.csv')
train.head()

#first 5 rows of train dataset
train.shape,test.shape
#Structure of train dataset

train.info()
#structure of test dataset

test.info()

#Item_Outlet_Sales is missing

#Item_Weight and Outlet_Size has missing data in both datasets
#combine train and test data 

complete = pd.concat([train, test],ignore_index=True)

complete

#find missing values

complete.isnull().sum()
#summary of dataset

complete.describe(include = 'all')
#unique values in each column



complete['Item_Identifier'].value_counts()

#so there are 1559 products
complete['Outlet_Identifier'].value_counts()

#10 Outlets
for col in ['Item_Type','Outlet_Type','Outlet_Location_Type','Outlet_Size','Item_Fat_Content']:

    print (complete[col].value_counts())

    

#Item_type :some groups can be combined

#Item_Fat_Content :Low Fat and Regular are the only two categories.
complete['Item_Fat_Content'] = complete['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
complete.Item_Fat_Content.value_counts()
#non consumable items should have a seperate category in Item_Fat_content

complete.loc[complete.Item_Type.isin(['Health and Hygiene', 'Household', 'Others']),'Item_Fat_Content'] = 'None'
complete.Item_Fat_Content.value_counts()
#create new Item type based on first two characters of Item ID

complete['Item_Type_Broad'] = complete['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:

complete['Item_Type_Broad'] = complete['Item_Type_Broad'].map({'FD':'Food',

                                                             'NC':'Non-Consumable',

                                                             'DR':'Drinks'})

complete['Item_Type_Broad'].value_counts()
#Plotting histogram:

complete['Item_Weight'].hist(color='white', edgecolor='k')

plt.title("True valued Histogram")

plt.xlabel("X-axis")

plt.ylabel("Item_Weight")

plt.show()
complete_mean = complete.copy(deep= True)
complete_mean['Item_Weight'].fillna(value=12.792854 , inplace=True)

#fill missing values with column mean
#plotting histogram of the new data

complete_mean['Item_Weight'].hist(color='white', edgecolor='k')

plt.title("Histogram with mean values")

plt.xlabel("X-axis")

plt.ylabel("Item_Weight")

plt.show()

#Histogram is different from original data

#we will replace missing values with mean weight of each item
ItemMeanWeight = complete.groupby('Item_Identifier').Item_Weight.mean()

complete.Item_Weight.fillna(0, inplace = True)

for index, row in complete.iterrows():

    if(row.Item_Weight == 0):

        complete.loc[index, 'Item_Weight'] = ItemMeanWeight[row.Item_Identifier]



#plot histogram of new dataset

complete['Item_Weight'].hist(color='white', edgecolor='k')

plt.xlabel("X-axis")

plt.ylabel("Item_Weight")

plt.show()

#the histogram is reasonably simillar to the original
#create column based on Ite_MRP

complete['MRP_Factor'] = pd.cut(complete.Item_MRP, [0,70,130,201,400], labels=['Low','Medium','High','Very High'])
complete.groupby('Outlet_Identifier').Outlet_Size.value_counts(dropna=False)


complete.groupby('Outlet_Type').Outlet_Size.value_counts(dropna=False)



#replace nan in grocery store and SM type 1 with small

complete.loc[complete.Outlet_Identifier.isin(['OUT010','OUT017','OUT045']), 'Outlet_Size'] = 'Small'



#creating dummy variables for categorical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



complete['Outlet'] = le.fit_transform(complete['Outlet_Identifier'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Broad','Outlet_Type','Outlet', 'MRP_Factor']

le = LabelEncoder()

for i in var_mod:

    complete[i] = le.fit_transform(complete[i])

    

complete = pd.get_dummies(complete, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type',

                              'Item_Type_Broad','Outlet', 'MRP_Factor'])
#change outlet establishment year to reflect how old an outlet is.

complete.Outlet_Establishment_Year = 2013-complete.Outlet_Establishment_Year

#drop item type

complete.drop(['Item_Type'],axis=1,inplace=True)

#seperating train and test datasets

Train = complete[complete.Item_Outlet_Sales.notnull()].copy(deep=True)

Test = complete[complete.Item_Outlet_Sales.isnull()].copy(deep=True)
Test.corr()

#there is no correlation between predictor vars
#QQ plot

quantile = Train.Item_Outlet_Sales



stats.probplot(quantile, dist="uniform", plot=pylab)

pylab.show()



#graph is almost linear
#residual vs fitted plot

from seaborn import residplot

residplot('Item_Visibility', 'Item_Outlet_Sales', train)

#no pattern 
#creating X and y variables

X = Train.copy(deep=True)

X.drop(['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier'],axis=1,inplace=True)

y = Train.copy(deep=True)

y = y['Item_Outlet_Sales']
#splitting into test and train data

from sklearn.cross_validation import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X,y,test_size = .2,random_state =0)

#linear regression model

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X_train,y_train)



y_pred = reg.predict(X_test)

print(reg.score(X_train,y_train))
#rmse value for linear reg model

from sklearn.metrics import mean_squared_error 

from math import sqrt



rmse = sqrt(mean_squared_error(y_test,y_pred))

rmse

#Cross validation with k=10 folds

from sklearn.model_selection import cross_val_predict, cross_val_score

ypred=cross_val_predict(reg, X_train, y_train, cv=10)

print(ypred)

#plotting coef of lin reg model

coef = pd.Series(reg.coef_, X_train.columns).sort_values()

coef.plot(kind='bar', title='Model Coefficients', figsize=(10,6))
#ridge reg model

from sklearn.linear_model import Ridge

ridgereg = Ridge(alpha=.05,normalize=True)

ridgereg.fit(X_train, y_train)
y_pred = ridgereg.predict(X_test)

y_pred
ridgereg.score(X_test, y_test)
#rmse value for ridge reg model



rmse = sqrt(mean_squared_error(y_test,y_pred))

rmse

#rmse value is almost unchanged
#plotting coef of ridge reg model

coef = pd.Series(ridgereg.coef_, X_train.columns).sort_values()

coef.plot(kind='bar', title='Model Coefficients', figsize=(10,6))
# Lasso Rigression:

from sklearn.linear_model import Lasso

lreg = Lasso(alpha=.05,normalize=True, max_iter=1e5)

lreg.fit(X_train, y_train)
y_pred = lreg.predict(X_test)

y_pred
#rmse value for lasso reg model



rmse = sqrt(mean_squared_error(y_test,y_pred))

rmse

#rmse value is almost unchanged
#plotting coef of lasso reg model

coef = pd.Series(lreg.coef_, X_train.columns).sort_values()

coef.plot(kind='bar', title='Model Coefficients', figsize=(10,6))
#since lasso reg has lowest rmse use it to predict test data

X_test = Test.copy(deep=True)
X_test.drop(['Item_Identifier','Item_Outlet_Sales','Outlet_Identifier'],axis=1,inplace=True)

y_pred = lreg.predict(X_test)

y_pred.shape