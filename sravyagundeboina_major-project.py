# importing data

import sys

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from datetime import datetime

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor





if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
# Reading Data

covid=pd.read_csv('/kaggle/input/carona/covid-data.csv')

covid
cd=covid[(covid.location=='India')]

cd
# Drawing Histograms

fig=plt.figure(figsize=(12,12))

ax=fig.gca()

cd.hist(ax=ax)

fig.tight_layout(pad=1)

plt.show()
# Mean

print("MEAN FOR EACH COLUMN :\n",np.mean(cd,axis=0))
# Median

print("MEDIAN FOR EACH COLUMN :\n",cd.median(axis=0))
mode=cd.mode()

print("MODE FOR EACH COLUMN :\n",mode.iloc[0])
cd.dtypes
cd.isnull().sum()
#null values in numerical column, replace the null values by the mean

cd['total_tests'].fillna(cd['total_tests'].mean(),inplace=True)

cd['new_tests'].fillna(cd['new_tests'].mean(),inplace=True)

cd['total_tests_per_thousand'].fillna(cd['total_tests_per_thousand'].mean(),inplace=True)

cd['new_tests_per_thousand'].fillna(cd['new_tests_per_thousand'].mean(),inplace=True)

cd['new_tests_smoothed'].fillna(cd['new_tests_smoothed'].mean(),inplace=True)

cd['new_tests_smoothed_per_thousand'].fillna(cd['new_tests_smoothed_per_thousand'].mean(),inplace=True)

cd['stringency_index'].fillna(cd['stringency_index'].mean(),inplace=True)
# null values in categorical column, replace the null values by the mode

cd['tests_units'].fillna(cd['tests_units'].mode()[0],inplace=True)
cd.isnull().sum()
# Convert date column to ordinal

cd["date"]=pd.to_datetime(cd["date"])

cd["date"]=cd["date"].map(dt.datetime.toordinal)

cd
#  Drop all categorical columns

cd=cd.select_dtypes(exclude=['object'])

cd.info()
# SELECT total_cases AS TARGET VARIABLE

y=cd["total_cases"].values



#SELECT THE OTHER COLUMNS AS FEATURES

x=cd.drop(["total_cases"],axis=1)
# TRAIN-TEST-SPLIT

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5, random_state=40)
# LINEAR REGRESSION AND ACCURACY



# Create the regressor: reg_all

lin_reg= LinearRegression()



# Fit the regressor to the training data

lin_reg.fit(x_train,y_train)



# Predict on the test data: y_pred

y_pred = lin_reg.predict(x_test)









# ACCURACY

acc=lin_reg.score(x_test,y_test)

print('Percentage Accuracy :{:f} %'.format(100*acc))
#  RANDOM FOREST REGRESSOR AND ACCURACY

# Instantiate rf

rf = RandomForestRegressor()

            

# Fit rf to the training set    

rf.fit(x_train, y_train)

y_pred=rf.predict(x_test)











# ACCURACY

acc=rf.score(x_test,y_test)

print('Percentage Accuracy :{:f} %'.format(100*acc))
# CONVERTING 25 AUGUST 2020 DATE TO ORDINAL

date=datetime.strptime('2021/8/14','%Y/%m/%d')

datetime.toordinal(date)
#PREDICTING total_cases FOR A NEW DATE 

 # USING RANDOM FOREST REGRESSOR

rf.predict([[2737662,49148.0,40084.0,887.0,1237.092,156.919,160.451,4.425,5205175.111111111,

             193035.88486486487,6.2023247863247863,0.98915315315315317,334539.8606557377,

             1.17575409836065579,145.30900523560205,1980004385.0,850.41900000000004,78.2,14.989,

             6.4139999999999997,8626.674,91.2,1082.28,67.39,7.9,48.6,169.55,7.53,179.66]])
#USING LINEAR REGRESSION

lin_reg.predict([[2737662,49148.0,40084.0,887.0,1237.092,156.919,160.451,4.425,5205175.111111111,

                  193035.88486486487,6.2023247863247863,0.98915315315315317,334539.8606557377,

                  1.17575409836065579,145.30900523560205,1980004385.0,850.41900000000004,78.2,14.989,

                  6.4139999999999997,8626.674,91.2,1082.28,67.39,7.9,48.6,169.55,7.53,179.66]])