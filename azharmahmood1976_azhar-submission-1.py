import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_train.head()
df_train.describe()
df_train.info()
# TRAIN dataset has 81 colmun

print("train data shape",df_train.shape)
# Whereas TEST dataset has 80 colmun. Why 1 colmun less? becasue TEST dataset do not included the final sale price 

print("test data shape",df_test.shape)
# SalePrice is TARGET VARIABLE which need to predict

df_train.head()
# Duringthe DATA EXPLORATION phase, we need to plot data to visualize and 

# understand the distribution of data, to identify the outliers and to see the potential patterns

plt.style.use(style='ggplot')

plt.rcParams['figure.figsize'] = (14,12)
# Overview of the STATISTCS data of dataset colmuns

display(df_train.describe().transpose)
# We will develop plot for correlation matrix

corrmat = df_train.corr()
plt.figure(figsize = (14,12))

sns.heatmap(corrmat,square=True) 

plt.show()
# Referring to data set, average sale price of house is approx $180K with most of the sale price value inbetween $130K-214K

df_train.SalePrice.describe()
# To understand that which feature will have higher impact on SalePrice

plt.figure(figsize = (14,12))

corrmat ['SalePrice'].sort_values(ascending = True).plot(kind = 'bar')
# We need to identify the outliers. We identified 2 outliers on RIGHT BOTTOM side of below graph

# It will be good to remove a house > 4000 square foot area from the dataset

plt.figure(figsize=(14,12))

plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
df_train = df_train.drop(df_train[(df_train['GrLivArea']>3000)&(df_train['SalePrice']>60000)].index)
# We re-checked the status of data set after removingthe outliers

plt.figure(figsize=(14,12))

plt.scatter(df_train['GrLivArea'],df_train['SalePrice'])
print(df_train.shape)
df_train.corr()
# We need to check the corelation

plt.figure(figsize=(14,12))

corr_train = df_train.corr()

num=5

col = corr_train.nlargest(num,'SalePrice')['SalePrice'].index

coeff = np.corrcoef(df_train[col].values.T)

heatmp = sns.heatmap(coeff,annot = True, xticklabels = col.values, yticklabels = col.values, linewidth = 4, cmap='PiYG', linecolor='blue')
# In order to perform regression, it make sense to log transform the target variable subjected if it is SKEWED

# One objective of this process is to improve the linearity

print("Skew is:", df_train.SalePrice.skew())
df_train.isna()
df_train.isna().any()
sum(df_train.isna().any())
import seaborn as sns
import os
df_train['GarageQual'].value_counts()
list_obj_col = df_train.select_dtypes(include='object')
list_obj_col = df_train.select_dtypes(include='object').columns
list_obj_col.shape
list_obj_col
list_num_col = list(df_train.select_dtypes(exclude='object').columns)
list_num_col
def fillna_all(df):

    for col in list_obj_col:

        df[col].fillna(value=df[col].mode()[0],inplace=True)

    for col in list_num_col:

        df[col].fillna(value=df[col].mean(),inplace=True)
fillna_all(df_train)
df_train.info()
df_train.isna().any()
sum(df_train.isna().any())
df_train.info()
for col in list_obj_col:

    print(col,'-',df_train[col].nunique())
temp = df_train['Id']

dummy = pd.get_dummies(df_train[list_obj_col],prefix=list_obj_col)
dummy
df_train.drop(list_obj_col,axis=1,inplace=True)
df_train.shape
df_train_final = pd.concat([df_train,dummy],axis=1)
dummy.shape
df_train_final.shape
df_train_final
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.info()
df_test.info()
fillna_all(df_test)
dummy1 = pd.get_dummies(df_test[list_obj_col],prefix=list_obj_col)
dummy1.shape
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
list_num_col+list_obj_col
df_train_test = pd.concat([df_train.drop('SalePrice',axis=1),df_test],axis=0)
df_train_test.shape
fillna_all(df_train_test)
df_train_test.info()
dummy3 = pd.get_dummies(df_train_test[list_obj_col], prefix=list_obj_col)
dummy.shape,dummy1.shape
dummy3.shape
df_train_test.drop(list_obj_col,axis=1,inplace=True)
df_train_test.shape
df_train_test_final = pd.concat([df_train_test,dummy3],axis=1)
df_train_test_final.shape
df_train_test_final.head()
X_train = df_train_test_final.iloc[0:1460]

X_test = df_train_test_final.iloc[1460:]
X_train.shape,X_test.shape
y=df_train['SalePrice']
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=23)
model.fit(X_train,y)
y_predict = model.predict(X_test)
y_predict
df_submission = pd.DataFrame({'Id':df_test['Id'],'SalePrice':y_predict})
df_submission
df_submission.to_csv('sub2.csv',index=False)