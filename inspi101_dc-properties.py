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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
dcp=pd.read_csv("../input/DC_Properties.csv")
#Splitting the dataset into known and unknown Price data

unknown_data=dcp[dcp['PRICE'].isnull()]

df=dcp[dcp['PRICE'].notnull()]
#Elimination of redundant variables

df=df.drop(['Unnamed: 0','YR_RMDL',"X","Y",'QUADRANT','SQUARE','CENSUS_TRACT','CENSUS_BLOCK','ASSESSMENT_SUBNBHD','LATITUDE','LONGITUDE','NATIONALGRID','ZIPCODE','CITY','STATE','CMPLX_NUM','GIS_LAST_MOD_DTTM','AYB','QUALIFIED','BLDG_NUM','GRADE','USECODE','FULLADDRESS','LIVING_GBA','WARD'],axis=1)
#Check for the missing values

df.apply(lambda x: x.count(), axis=0)

df.isnull().sum(axis=0)
#Checking and replacing outliers

sns.set(style="whitegrid")

ax = sns.boxplot(x='STORIES', data=df, fliersize=15, whis=2.5)

plt.show()
df1=df[df['STORIES'].notnull()]

df2=df1[df1['STORIES']<=df1['STORIES'].quantile(0.9999)]
#Replacing the missing values of numeric variables

df['NUM_UNITS'].fillna(df['NUM_UNITS'].mean(),inplace=True)

df.apply(lambda x: x.count(), axis=0)
df['STORIES'].fillna(df2['STORIES'].mean(),inplace=True)

df['GBA'].fillna(df['GBA'].mean(),inplace=True)

df['KITCHENS'].fillna(round(df['KITCHENS'].mean(),0),inplace=True)
#Converting the format to timeseries

df['SALEYEAR']=df['SALEDATE'].str[0:4]

df['SALEDATE'].fillna(value='2019-01-10 00:00:00',inplace=True)

df['SALEYEAR'] = pd.to_datetime(df['SALEDATE'])

df['SALEYEAR'] = pd.DatetimeIndex(df['SALEYEAR']).year
#Calculating the age of the house at time of sale

df['AGE'] =df['SALEYEAR']-df['EYB']
df.drop(['EYB','SALEDATE'], axis=1, inplace=True)
#Converting AC into binary

df.AC.replace('0', 0, inplace = True)

df.AC.replace('Y', 1, inplace = True)

df.AC.replace('N', 0, inplace = True)
#Transforming categorical variables into dummies with binary values

number=LabelEncoder()

df['HEAT']=number.fit_transform(df.HEAT.astype(str))

df['CNDTN']=number.fit_transform(df.CNDTN.astype(str))

df['ROOF']=number.fit_transform(df.ROOF.astype(str))

df['STRUCT']=number.fit_transform(df.STRUCT.astype(str))

df['STYLE']=number.fit_transform(df.STYLE.astype(str))

df['EXTWALL']=number.fit_transform(df.EXTWALL.astype(str))

df['INTWALL']=number.fit_transform(df.INTWALL.astype(str))

df['SOURCE']=number.fit_transform(df.SOURCE.astype(str))

df['ASSESSMENT_NBHD']=number.fit_transform(df.ASSESSMENT_NBHD.astype(str))
#Normalizing the data

from scipy import stats

df=df[(np.abs(stats.zscore(df.PRICE)) < 6)]
#Re-evaluating the missing values

price_known=df[df['PRICE'].notnull()]

price_known.isnull().sum(axis=0)
#Applying Linear Regression with train test split of 80-20 and creating the model

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
features = price_known.drop('PRICE', axis=1)

target = price_known.PRICE
X_train, X_test, Y_train, Y_test = train_test_split(features,target,test_size=0.2, random_state=42)
#import statsmodels.api as sm

#model = sm.OLS(Y_train,X_train.astype(float)).fit()

#print(model.summary())
#Calculating RMSE to check for model accuracy

#from sklearn import metrics

#predictions=model.predict(X_test)

#print('Model Performance')

#print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))