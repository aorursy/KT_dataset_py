# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(r'/kaggle/input/insurance/insurance.csv')
df.dtypes
df['age'].isna().value_counts()

df.columns
def checkNull (data) :

    columns = data.columns

    for column in columns :

        print('****Checking null values for '+ str(column) + ' ****')

        print(data[column].isna().value_counts())

        print('****Checking competed*****')

    
checkNull(df)
corrMatrix = df.corr()
fig = plt.figure(figsize = (8,6))

ax= fig.add_subplot(111)

sns.heatmap(corrMatrix,annot=True,ax=ax)
df[['sex','charges']].plot(kind = 'scatter',x='sex',y='charges')
df['region'].value_counts()
df[df['region'] == 'southeast'][['region','charges']].plot(kind='hist',title = 'southeast')
df[df['region'] == 'southwest'][['region','charges']].plot(kind='hist',title = 'southwest')
df[df['region'] == 'northwest'][['region','charges']].plot(kind='hist',title = 'northwest')
df[df['region'] == 'northeast'][['region','charges']].plot(kind='hist',title = 'northeast')
df_smoking = df[['smoker','charges']]

#df_smoker = df_smoking[df_smoking['smoker'] == 'yes']

df_smoking['smoker'].value_counts()
df_smoking.replace({'yes' : 1,'no': 0},inplace = True)
fig1 = plt.figure(figsize = (12,6))

fig1.subplots_adjust(wspace = 0.5)

ax1 = fig1.add_subplot(121)

sns.boxplot(x='smoker',y='charges',data=df_smoking,palette='rainbow',ax =ax1,showfliers=False)

ax1.set_xticklabels(['Non-Smoker','Somker'])

ax1.set_xlabel('')

ax1.set_ylabel('Insurance Charges')



ax1 = fig1.add_subplot(122) 

sns.violinplot(data =df_smoking, x='smoker', y='charges',ax= ax1)

ax1.set_xticklabels(['Non-Smoker','Somker'])

ax1.set_xlabel('')

ax1.set_ylabel('Insurance Charges')
mat = df_smoking.corr()

sns.heatmap(mat,annot=True)
df_gen = df[['sex','charges']]

fig2 = plt.figure(figsize = (15,6))

fig2.subplots_adjust(wspace = 0.5)

ax2 = fig2.add_subplot(121) 

sns.boxplot(data= df_gen,x = 'sex',y='charges',showfliers=False,ax=ax2)

ax2 = fig2.add_subplot(122)

sns.violinplot(data =df_gen, x='sex', y='charges',ax=ax2)
#df_gen.replace({'male': 1,'female':0},inplace = True)

df_gen.corr()
df_model =                  df.replace({

                                        'southeast':0,

                                        'northwest':1,

                                        'southwest':2,

                                        'northeast':3,

                                        'male': 1,

                                        'female':0,

                                        'yes' : 1,

                                        'no': 0

                                         })

df_model.head()
figg = plt.figure(figsize = (10,8))

ax3 = figg.add_subplot(111) 

sns.heatmap(df_model.corr(),annot=True,ax=ax3)
total_data_count = df_new_model.shape[0]

train_data_index = round(total_data_count*0.8) -1

train_data = df_new_model.loc[:train_data_index,:]

test_data_index = train_data_index +1

test_data = df_new_model.loc[test_data_index:,:]
test_data.shape

train_data.shape
from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

#reg = linear_model.LinearRegression() 

#x = df_new_model[['age','sex','bmi','children','smoker','region']]

#x = df_model[['smoker']]

#y = df_new_model[['charges']]

#reg.fit(x,y)

#print('The coefficients are ',reg.coef_)

#print('The intercept is ',reg.intercept_)
y_hat = reg.predict(test_data[['age','sex','bmi','children','smoker','region']])

print ('R2 score is ',r2_score(test_data['charges'],y_hat))

print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))
df_new_model = pd.get_dummies(df,columns=['smoker','sex','region'],prefix =['smoker','sex','region'])

df_new_model.head()

#df_new_model.columns
fig_new = plt.figure(figsize = (10,8))

ax_new = fig_new.add_subplot(111) 

sns.heatmap(df_new_model.corr(),annot=True,ax=ax_new)
total_data_count = df_new_model.shape[0]

train_data_index = round(total_data_count*0.8) -1

train_data = df_new_model.loc[:train_data_index,:]

test_data_index = train_data_index +1

test_data = df_new_model.loc[test_data_index:,:]
from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

reg = linear_model.LinearRegression() 

x = df_new_model[['age', 'bmi', 'children','smoker_no', 'smoker_yes',

       'sex_female', 'sex_male', 'region_northeast', 'region_northwest',

       'region_southeast', 'region_southwest']]

y = df_new_model[['charges']]

reg.fit(x,y)

print('The coefficients are ',reg.coef_)

print('The intercept is ',reg.intercept_)
y_hat = reg.predict(test_data[['age', 'bmi', 'children','smoker_no', 'smoker_yes',

       'sex_female', 'sex_male', 'region_northeast', 'region_northwest',

       'region_southeast', 'region_southwest']])

print ('R2 score is ',r2_score(test_data['charges'],y_hat))

print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))
total_data_count = df_new_model[['smoker_yes']].shape[0]

train_data_index = round(total_data_count*0.8) -1

train_data = df_new_model.loc[:train_data_index,:]

test_data_index = train_data_index +1

test_data = df_new_model.loc[test_data_index:,:]
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



def ployFuntion (degrees,train_data,test_data) :

    for degree in degrees :

        poly = PolynomialFeatures(degree)

        train_x_poly = poly.fit_transform(train_data[['age','bmi','smoker_no', 'smoker_yes']])

        test_x_poly = poly.fit_transform(test_data[['age','bmi','smoker_no', 'smoker_yes']])

        reg = linear_model.LinearRegression()

        reg.fit(train_x_poly,train_data['charges'])

        y_hat = reg.predict(test_x_poly)

        print('Degree of polynomial model used :',degree)

        print ('R2 score is ',r2_score(test_data['charges'],y_hat))

        print('RMSE score is ',math.sqrt(mean_squared_error(test_data['charges'],y_hat)))

        print ('MAE score is ',mean_absolute_error(test_data['charges'],y_hat))

        print ('*****************************')

ployFuntion(degrees = list(range(2,12)),train_data = train_data,test_data =test_data)
from sklearn.preprocessing import StandardScaler

stan = StandardScaler()

std_charges = stan.fit_transform(df_new_model[['charges']])
from sklearn.preprocessing import StandardScaler

stan = StandardScaler()

std_bmi = stan.fit_transform(df_new_model[['bmi']])
from sklearn.preprocessing import StandardScaler

stan = StandardScaler()

std_age = stan.fit_transform(df_new_model[['age']])
df_new_model['age'] = std_age

df_new_model['bmi'] = std_bmi

df_new_model['charges'] = std_charges