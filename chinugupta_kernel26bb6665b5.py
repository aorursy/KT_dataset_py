# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import sys

import scipy.stats as ss

import sklearn.preprocessing as skpe

import sklearn.model_selection as ms

import sklearn.metrics as sklm

import sklearn.linear_model as lm

import sklearn.ensemble as sken

import matplotlib.pyplot as plt

import seaborn as sns
# Data Description

"""The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA. The following describes the dataset columns:



    CRIM - per capita crime rate by town

    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.

    INDUS - proportion of non-retail business acres per town.

    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)

    NOX - nitric oxides concentration (parts per 10 million)

    RM - average number of rooms per dwelling

    AGE - proportion of owner-occupied units built prior to 1940

    DIS - weighted distances to five Boston employment centres

    RAD - index of accessibility to radial highways

    TAX - full-value property-tax rate per $10,000

    PTRATIO - pupil-teacher ratio by town

    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town

    LSTAT - % lower status of the population

    MEDV - Median value of owner-occupied homes in $1000's"""

path="../input/boston-housing-dataset/HousingData.csv"

df=pd.read_csv(path)
df.shape
df.columns
# Variable Identification

df.dtypes
df.info()
# Continuous Varaible

df.describe()    # CHAS seems to be a categorical variable from the description and describe method
# Categorical Variable

(df['CHAS'].value_counts()/len(df['CHAS'])*100).plot.bar()   # Frequency Chart
# Checking correlation bw different variables

plt.figure(figsize=(18,18))

sns.heatmap(df.corr(),vmax=.7,cbar=True,annot=True)   # From this heatmap we can conclude that INDUS, NOX AND AGE are highly correlated with DIS.Also the target variable MEDV is having a correlation score > 0.4 with following features:LSTAT, PTRATIO, RM, TAX, INDUS and NOX  
scaler = skpe.MinMaxScaler()

column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']

x = df.loc[:,column_sels]

y = df['MEDV']

x = pd.DataFrame(data=scaler.fit_transform(x), columns=column_sels)

fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for i, k in enumerate(column_sels):

    sns.scatterplot(y=y, x=x[k], ax=axs[i])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
sns.scatterplot(x='LSTAT',y='INDUS',data=df)
sns.scatterplot(x='LSTAT',y='RM',data=df)
sns.scatterplot(x='LSTAT',y='AGE',data=df)
sns.scatterplot(x='TAX',y='INDUS',data=df)
sns.scatterplot(x='TAX',y='NOX',data=df)
sns.scatterplot(x='TAX',y='LSTAT',data=df)
sns.scatterplot(x='NOX',y='INDUS',data=df)
# Box Plots to detect outliers

fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in df.items():

    sns.boxplot(y=k, data=df, ax=axs[index])

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)  # Columns like CRIM, ZN, RM and B seems to have outliers
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))

index = 0

axs = axs.flatten()

for k,v in df.drop(['CHAS'],axis=1).items():

    sns.distplot(v, ax=axs[index])

    index += 1

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)   # We have dropped CHAS for this analysis as it is a discrete variable
print(df['AGE'].skew())

sns.distplot(np.log1p(df['AGE']),color='Black')

print(np.log1p(df['AGE']).skew())
print(df['B'].skew())

sns.distplot(np.power(df['B'],3),color='Black')

print(np.power(df['B'],3).skew())
print(df['PTRATIO'].skew())

sns.distplot(np.sqrt(df['PTRATIO']),color='Black')

print(np.sqrt(df['PTRATIO']).skew())
# Removing skewness of these features

df['LSTAT']=np.log1p(df['LSTAT'])

df['AGE']=np.log1p(df['AGE'])

df['PTRATIO']=np.log1p(df['PTRATIO'])

df['B']=np.power(df['B'],3)
df.isnull().sum()
# Replacing missing values with the Mean

avg_mean_cr = df['CRIM'].astype('float').mean(axis=0)

df['CRIM'].replace(np.nan,avg_mean_cr,inplace=True)



avg_mean_zn = df['ZN'].astype('float').mean(axis=0)

df['ZN'].replace(np.nan,avg_mean_zn,inplace=True)



avg_mean_in = df['INDUS'].astype('float').mean(axis=0)

df['INDUS'].replace(np.nan,avg_mean_in,inplace=True)



avg_mean_ch = df['CHAS'].astype('float').mean(axis=0)

df['CHAS'].replace(np.nan,avg_mean_ch,inplace=True)



avg_mean_ls = df['LSTAT'].astype('float').mean(axis=0)

df['LSTAT'].replace(np.nan,avg_mean_ls,inplace=True)



avg_mean_age = df['AGE'].astype('float').mean(axis=0)

df['AGE'].replace(np.nan,avg_mean_age,inplace=True)
# Segregating features and labels and dropping unnecessary features

x=df.drop(['MEDV'],axis=1)

y=df['MEDV']
# Scaling values using MinMaxScaler

scaler=skpe.MinMaxScaler()

x_scaled=scaler.fit_transform(x)

x=pd.DataFrame(x_scaled,columns=x.columns)

x.head()
# Splitting dataset into train and test sets

train_x,test_x,train_y,test_y=ms.train_test_split(x,y,test_size=0.2,random_state=115)

train_x.shape,test_x.shape
lr=lm.LinearRegression(normalize=True)

lr.fit(train_x,train_y)

lr.coef_
import statsmodels.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std

model=sm.OLS(train_y,train_x)

result=model.fit()

result.summary()
# Checking the probabality distribution of the two quantiles



# Q-Q Normal plot

def resid_qq(y_test, y_score):

    ## first compute vector of residuals. 

    resids = np.subtract(test_y, y_score)

    ## now make the residual plots

    ss.probplot(resids, plot = plt)

    plt.title('Residuals vs. predicted values')

    plt.xlabel('Quantiles of standard Normal distribution')

    plt.ylabel('Quantiles of residuals')

    

y_score=lr.predict(test_x)

resid_qq(test_y, y_score)         # This is perfectly fine even though it has a few outliers
# A common misconception is that the features or label of a linear regression model must have Normal distributions. This is not the case! Rather, the residuals (errors) of the model should be Normally distributed



def hist_resids(y_test, y_score):

    ## first compute vector of residuals. 

    resids = np.subtract(test_y, y_score)

    ## now make the residual plots

    sns.distplot(resids)

    plt.title('Histogram of residuals')

    plt.xlabel('Residual value')

    plt.ylabel('count')

    

hist_resids(test_y, y_score)        # This is almost good exvept for the reason that it is slightly right-skewed
# Creating a dataframe of residuals,test values and predicted values

residuals = pd.DataFrame({

    'fitted values' : test_y,

    'predicted values' : y_score

})



residuals['residuals'] = residuals['fitted values'] - residuals['predicted values']

residuals.head()
# Residual scatterplot

# Remember,there should be no pattern observed in this plot;the residuals should be randomly distributed across the plot

f = range(0,102)

k = [0 for i in range(0,102)]

plt.scatter( f, residuals.residuals[:], label = 'residuals')

plt.plot( f, k , color = 'red', label = 'regression line' )

plt.xlabel('fitted points ')

plt.ylabel('residuals')

plt.title('Residual plot')

plt.ylim(-4,4)

plt.legend()      # This is good
# Distribution of Error terms

plt.hist(residuals.residuals, bins = 150)

plt.xlabel('Error')

plt.ylabel('Frequency')

plt.title('Distribution of Error Terms')

plt.show()        # It seems fine but there are a few outliers
# Segregating features and labels

x=df.drop(['MEDV'],axis=1)

y=df['MEDV']
Coefficients = pd.DataFrame({

    'Variable'    : x.columns ,

    'coefficient' : lr.coef_

})

Coefficients.head()
x = range(len(train_x.columns))

y = lr.coef_

plt.bar( x, y )

plt.xlabel( "Variables")

plt.ylabel('Coefficients')

plt.title('Normalized Coefficient plot')