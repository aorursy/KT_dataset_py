# Importing Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import os



# for train-test split of dataset

from sklearn.model_selection import train_test_split



# for scaling of dataset

from sklearn.preprocessing import MinMaxScaler



# RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



# to create linear model

import statsmodels.api as sm



# to check VIFs

from statsmodels.stats.outliers_influence import variance_inflation_factor



# R-squared

from sklearn.metrics import r2_score



# MSE

from sklearn.metrics import mean_squared_error



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
print(os.listdir('../input/geely-auto'))
# importing csv file

car_df = pd.read_csv("../input/geely-auto/CarPriceAssignment.csv")

car_df.head()
print(car_df.shape)

car_df.info()
# Describing the numerical vars

car_df.describe()
sns.pairplot(car_df)

plt.show()
# searching for outliers in compressionratio



# Box-plot

plt.figure(figsize=(15,6))

plt.subplot(121)

sns.boxplot(data=car_df.compressionratio, width=0.5, palette="colorblind")

plt.title('Box-Plot Compression Ratio')



# Scatter plot

plt.subplot(122)

sns.scatterplot(data=car_df.compressionratio, palette="colorblind")

plt.title('Scatter Plot Compression Ratio')

plt.show()
# percentage of outliers

comp_ratio = car_df[['compressionratio']].copy().sort_values(by='compressionratio',ascending=False)

comp_ratio_outlier = car_df[car_df['compressionratio']>12]

print(len(comp_ratio))

print(len(comp_ratio_outlier))



comp_ratio_outlier_perc = round(100*(len(comp_ratio_outlier) / len(comp_ratio)),2)

print('Outlier percentage of compressionratio: ' + str(comp_ratio_outlier_perc))
# heatmap

plt.figure(figsize = (20,10))  

sns.heatmap(car_df.corr(), cmap= 'YlGnBu',annot = True)
# Keeping 'citympg' and 'carlength' from above

car_df = car_df.drop(['carwidth','curbweight','wheelbase','highwaympg'], axis=1)

car_df.head()
# dropping car_ID

car_df = car_df.drop('car_ID', 1)

car_df.head()
# Categorical columns

car_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
plt.figure(figsize=(20, 20))

plt.subplot(3,3,1)

sns.boxplot(x = 'fueltype', y = 'price', data = car_df)

plt.subplot(3,3,2)

sns.boxplot(x = 'aspiration', y = 'price', data = car_df)

plt.subplot(3,3,3)

sns.boxplot(x = 'doornumber', y = 'price', data = car_df)

plt.subplot(3,3,4)

sns.boxplot(x = 'carbody', y = 'price', data = car_df)

plt.subplot(3,3,5)

sns.boxplot(x = 'drivewheel', y = 'price', data = car_df)

plt.subplot(3,3,6)

sns.boxplot(x = 'enginelocation', y = 'price', data = car_df)

plt.subplot(3,3,7)

sns.boxplot(x = 'enginetype', y = 'price', data = car_df)

plt.subplot(3,3,8)

sns.boxplot(x = 'cylindernumber', y = 'price', data = car_df)

plt.subplot(3,3,9)

sns.boxplot(x = 'fuelsystem', y = 'price', data = car_df)

plt.show()
car_df.CarName.head(20)
# split and taking the Company name

car_df['CarName'] = car_df['CarName'].apply(lambda x:re.split('-| ',x)[0])

car_df['CarName'].head(10)
# Names and count of Cars according to Companies

car_df['CarName'].value_counts()
# mapping similar companies into one

car_df['CarName'] = car_df.CarName.str.replace('vw','volkswagen')

car_df['CarName'] = car_df.CarName.str.replace('vokswagen','volkswagen')

car_df['CarName'] = car_df.CarName.str.replace('toyouta','toyota')

car_df['CarName'] = car_df.CarName.str.replace('porcshce','porsche')

car_df['CarName'] = car_df.CarName.str.replace('maxda','mazda')

car_df['CarName'] = car_df.CarName.str.replace('Nissan','nissan')

car_df['CarName'].value_counts()
print(car_df['fueltype'].value_counts())

print(car_df['aspiration'].value_counts())

print(car_df['doornumber'].value_counts())

print(car_df['enginelocation'].value_counts())
# quantifying into 1 and 0

car_df['fueltype'] = car_df['fueltype'].map({'gas': 1, 'diesel':0})

car_df['aspiration'] = car_df['aspiration'].map({'std': 1, 'turbo':0})

car_df['doornumber'] = car_df['doornumber'].map({'four': 1, 'two':0})

car_df['enginelocation'] = car_df['enginelocation'].map({'front': 1, 'rear':0})



car_df.head()
# creating dummy variables for categorical columns

dummy_car_df = pd.get_dummies(car_df, drop_first=True)

dummy_car_df.head()
dummy_car_df.info()
# train-test split

np.random.seed(0)

df_train, df_test = train_test_split(dummy_car_df, train_size = 0.7, random_state=100)

print(df_train.shape)

df_train.head()
# Apply scalar to all columns except 'quantified' and 'dummy' variables

vars_list = ['price','carlength','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg']



scalar = MinMaxScaler()

df_train[vars_list] = scalar.fit_transform(df_train[vars_list])

df_train.head()
df_train.describe()
y_train = df_train.pop('price')

X_train = df_train

print(y_train.head())

X_train.head()
# RFE

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm,12)      # choosing top 12 features

rfe = rfe.fit(X_train,y_train)

list(zip(X_train.columns, rfe.support_, rfe.ranking_))
# selecting the top 12 features

col = X_train.columns[rfe.support_]

col
# variables which are redundant

X_train.columns[~rfe.support_]
# eatures selected for model building

X_train_rfe = X_train[col]

X_train_rfe.head()
# add constant variable

X_train_rfe_1 = sm.add_constant(X_train_rfe)



# 1st linear model

lr_model_1 = sm.OLS(y_train,X_train_rfe_1).fit()



# summary

lr_model_1.summary()
# VIF of model_1

vif = pd.DataFrame()

X = X_train_rfe_1.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# heatmap

plt.figure(figsize=(15,8))

sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)
# dropping 'enginetype_rotor' 

X_train_rfe_2 = X_train_rfe_1.drop('enginetype_rotor', axis=1)
# add constant variable

X_train_rfe_2 = sm.add_constant(X_train_rfe_2)



# 2nd linear model

lr_model_2 = sm.OLS(y_train,X_train_rfe_2).fit()



# summary

lr_model_2.summary()
# VIF of model_2

vif = pd.DataFrame()

X = X_train_rfe_2.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# dropping 'cylindernumber_four'

X_train_rfe_3 = X_train_rfe_2.drop('cylindernumber_four',1)



# add constant variable

X_train_rfe_3 = sm.add_constant(X_train_rfe_3)



# 3rd linear model

lr_model_3 = sm.OLS(y_train,X_train_rfe_3).fit()



# summary

lr_model_3.summary()
# VIF of model_3

vif = pd.DataFrame()

X = X_train_rfe_3.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# dropping 'boreratio'

X_train_rfe_4 = X_train_rfe_3.drop('boreratio',1)



# add constant variable

X_train_rfe_4 = sm.add_constant(X_train_rfe_4)



# 4th linear model

lr_model_4 = sm.OLS(y_train,X_train_rfe_4).fit()



# summary

lr_model_4.summary()
# VIF of model_4

vif = pd.DataFrame()

X = X_train_rfe_4.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# heatmap

plt.figure(figsize=(12,5))

sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)
# dropping 'carlength'

X_train_rfe_5 = X_train_rfe_4.drop('carlength',1)



# add constant variable

X_train_rfe_5 = sm.add_constant(X_train_rfe_5)



# 5th linear model

lr_model_5 = sm.OLS(y_train,X_train_rfe_5).fit()



# summary

lr_model_5.summary()
# VIF of model_5

vif = pd.DataFrame()

X = X_train_rfe_5.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# dropping 'stroke'

X_train_rfe_6 = X_train_rfe_5.drop('stroke',1)



# add constant variable

X_train_rfe_6 = sm.add_constant(X_train_rfe_6)



# 6th linear model

lr_model_6 = sm.OLS(y_train,X_train_rfe_6).fit()



# summary

lr_model_6.summary()
# VIF of model_6

vif = pd.DataFrame()

X = X_train_rfe_6.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# dropping 'cylindernumber_three'

X_train_rfe_7 = X_train_rfe_6.drop('cylindernumber_three',1)



# add constant variable

X_train_rfe_7 = sm.add_constant(X_train_rfe_7)



# 7th linear model

lr_model_7 = sm.OLS(y_train,X_train_rfe_7).fit()



# summary

lr_model_7.summary()
# VIF of model_7

vif = pd.DataFrame()

X = X_train_rfe_7.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# heatmap

plt.figure(figsize=(8,5))

sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)
# dropping 'cylindernumber_twelve'

X_train_rfe_8 = X_train_rfe_7.drop('cylindernumber_twelve',1)



# add constant variable

X_train_rfe_8 = sm.add_constant(X_train_rfe_8)



# 8th linear model

lr_model_8 = sm.OLS(y_train,X_train_rfe_8).fit()



# summary

lr_model_8.summary()
# VIF of model_8

vif = pd.DataFrame()

X = X_train_rfe_8.drop('const',1)  # no need of 'const' in finding VIF

vif['features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'],2)

vif = vif.sort_values(by = 'VIF', ascending=False)

vif
# heatmap

plt.figure(figsize=(8,5))

sns.heatmap(X.corr(), cmap='YlGnBu', annot=True)
y_train_pred = lr_model_8.predict(X_train_rfe_8)

residual = y_train - y_train_pred



plt.figure()

sns.distplot(residual, bins = 20)

plt.title('Error Terms', fontsize = 18)     

plt.xlabel('Errors')   
fig = plt.figure(figsize=(18,5))

x_axis_range = [i for i in range(1,144,1)]

fig.suptitle('Error Terms', fontsize=20)



plt.subplot(1,2,1)

plt.scatter(x_axis_range, residual)

plt.ylabel('Residuals')



plt.subplot(1,2,2)

plt.plot(x_axis_range,residual, color="green", linewidth=2.5, linestyle="-")

vars_list = ['price','carlength','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg']



df_test[vars_list] = scalar.transform(df_test[vars_list])

df_test.head()
y_test = df_test.pop('price')

X_test = df_test

print(y_test.head())

X_test.head()
# creating X_test_new dataframe by selected columns from final model

X = X_train_rfe_8.drop('const',1)

X_test_new = X_test[X.columns]



# adding contant

X_test_new = sm.add_constant(X_test_new)

X_test_new.head()
# making prediction using final model

y_pred = lr_model_8.predict(X_test_new)

y_pred.head()
# R-squared value of test set

r2_score(y_test, y_pred)
np.sqrt(mean_squared_error(y_test, y_pred))
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure(figsize=(20,5))

fig.suptitle('y_test vs y_pred', fontsize=20)              



plt.subplot(1,2,1)

plt.scatter(y_test,y_pred)

plt.xlabel('y_test', fontsize=18)                          

plt.ylabel('y_pred', fontsize=16)                          



plt.subplot(1,2,2)

sns.regplot(y_test,y_pred)

plt.xlabel('y_test', fontsize=18)                          
