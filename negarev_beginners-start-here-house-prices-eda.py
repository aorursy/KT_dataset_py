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
# here are the modules we'll be using throughout this notebook

import statsmodels.api as sm

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

import math
from sklearn.metrics import mean_absolute_error

from scipy import stats
# Load our data from the csv file

houses = pd.read_csv('../input/housedata/data.csv') 
houses.shape
houses.dtypes
houses.country.value_counts()
houses.statezip.value_counts()
"The average price of a house is ${:,.0f}".format(houses.price.mean())
#get the average price for houses along their number of bedrooms:

plt.figure(figsize=(10,6))

sns.barplot(x=houses.bedrooms, y=houses['price'])
# get a price breakdown for each bedroom group

bybedroom = houses.groupby(['bedrooms']).price.agg([len, min, max])
#problem #1 and #2 - 2 houses with 0 bedrooms, giant outlier at 3 bedrooms

bybedroom
# problem #3 - houses with null prices

houses_zero= houses[houses.price==0]

print('There are '+str(len(houses_zero))+' houses without a price')
# problem #4 - house prices are not normal

sns.distplot(houses['price'], fit=norm)
# new dataframe without problem #1 #2 #3

houses_o = houses[(houses.price<2.5*10**7) & (houses.bedrooms>0) & (houses.price>0)].copy()
#recode houses with more than 6 bedrooms as 6 bedrooms

houses_o['bedrooms_recoded'] = houses_o['bedrooms'].replace([7,8,9],6)
houses_o['renovated_0_1'] = houses_o['yr_renovated']/houses_o['yr_renovated']

houses_o['renovated_0_1'] = houses_o['renovated_0_1'].fillna(0)
features = ['price','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',

       'floors', 'waterfront', 'view', 'condition', 'sqft_above',

       'sqft_basement', 'yr_built', 'yr_renovated']

mask = np.zeros_like(houses_o[features].corr(), dtype=np.bool) 

mask[np.triu_indices_from(mask)] = True 



f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix',fontsize=25)



sns.heatmap(houses_o[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 

            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
# Move our features into the X DataFrame

X = houses_o.loc[:,['bedrooms_recoded', 'floors','view','condition','renovated_0_1']]



# Move our labels into the y DataFrame

y = houses_o.loc[:,['price']] 
# separate y and X into train and test

X_train, X_test, y_train, y_test = train_test_split(

                                                    X, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=42

                                                   )
#train a basic multiple regression model and print out the coefficients

mod = sm.OLS(y_train, X_train)

res = mod.fit()

print(res.summary())
# Ask the model to predict prices in the train and test set based just on our predictor variables

lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)
# Now let's plot our predicted values on one axis and the real values on the other axis

plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")

plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper right")

plt.plot([0.2*10**6, 0.25*10**7], [0.2*10**6, 0.25*10**7], c = "red")

plt.show()
#get the results from the regression in dataframe format

res = pd.DataFrame(data=train_pre, columns=['predicted values'])

#join with the actual prices

res = y_train.reset_index().join(res)

#join with the training dataset

resfin = res.join(X_train, on='index',lsuffix='_y')

# compute the actual prices, predicted prices and error

resfin['predprice']=res['predicted values']

resfin['actprice']=res['price']

resfin['error']=resfin['predprice']-resfin['actprice']
#get the results from the regression in dataframe format

res_test = pd.DataFrame(data=test_pre, columns=['predicted values'])

#join with the actual prices

res_test = y_test.reset_index().join(res_test)

#join with the training dataset

resfin_test = res_test.join(X_test, on='index',lsuffix='_y')

# compute the actual prices, predicted prices and error

resfin_test['predprice']=resfin_test['predicted values']

resfin_test['actprice']=resfin_test['price']

resfin_test['error']=resfin_test['predprice']-resfin_test['actprice']

resdf = pd.concat([resfin,resfin_test])
"The mean error of our model is ${:,.0f}".format(resfin_test['error'].mean())

#plot the error

plt.figure(figsize=(15,8))

sns.distplot(resfin_test['error'], fit=norm)
#standardize the errors

x_array = np.array(resfin_test['error'])

normalized_X = stats.zscore(x_array)

#let's get the normalized error back into our dataset

error_df = pd.DataFrame(data=normalized_X.T, columns=['normalized error'])

resfin2 = resfin_test.join(error_df)

resfin2['abs_norm_error'] = abs(resfin2['normalized error'])

#now let's select only the errors that are 2 standard deviations away from the mean

resfin2['massive underestimation'] = resfin2['normalized error']<-2 

plt.figure(figsize=(10,5))

sns.distplot(error_df, fit=norm)
#how many big mistakes in our test dataset?

resfin2['massive underestimation'].value_counts()
"approximately {:.1%} of the test houses are massively underestimated".format(resfin2['massive underestimation'].values.sum()/len(resfin2))
plt.figure(figsize=(12,8))

plt.scatter(resfin2['predprice'], resfin2['actprice'], c = resfin2['massive underestimation'])

plt.plot([0.2*10**6, 1.75*10**6], [0.2*10**6, 1.75*10**6], c = "red")

plt.legend(loc = "upper left")
#Now let's explore - what kind of houses is the model particularly bad at estimating the price of?

pd.crosstab(resfin2['bedrooms_recoded'],resfin2['massive underestimation']).apply(lambda r: r/r.sum(), axis=1)
result = houses_o.groupby(["statezip"])['price'].aggregate(np.median).reset_index().sort_values('price', ascending=False)

plt.figure(figsize=(15,8))

chart = sns.barplot(

    x='statezip',

    y='price',

    data=houses_o,

    order = result['statezip'],

    estimator=np.median

    

    

)

chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
houses_o['posh_zip'] = houses_o['statezip'].isin(['WA 98039','WA 98004','WA 98040','WA 98109']).astype(int)

# Move our features into the X DataFrame

X = houses_o.loc[:,['bedrooms_recoded', 'floors', 'condition','view','renovated_0_1', 'posh_zip']]



# Move our labels into the y DataFrame

y = houses_o.loc[:,['price']] 
# separate y and X into train and test

X_train, X_test, y_train, y_test = train_test_split(

                                                    X, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=42

                                                   )
#train a basic multiple regression model and print out the coefficients

mod = sm.OLS(y_train, X_train)

res = mod.fit()

print(res.summary())
# Ask the model to predict prices in the train and test set based just on our predictor variables

lr = LinearRegression()

lr.fit(X_train,y_train)

test_pre = lr.predict(X_test)

train_pre = lr.predict(X_train)
#get the results from the regression in dataframe format

res_test = pd.DataFrame(data=test_pre, columns=['predicted values'])

#join with the actual prices

res_test = y_test.reset_index().join(res_test)

#join with the training dataset

resfin_test = res_test.join(X_test, on='index',lsuffix='_y')

# compute the actual prices, predicted prices and error

resfin_test['predprice']=resfin_test['predicted values']

resfin_test['actprice']=resfin_test['price']

resfin_test['error']=resfin_test['predprice']-resfin_test['actprice']

resdf = pd.concat([resfin,resfin_test])
#standardize the errors

x_array = np.array(resfin_test['error'])

normalized_X = stats.zscore(x_array)

#let's get the normalized error back into our dataset

error_df = pd.DataFrame(data=normalized_X.T, columns=['normalized error'])

resfin2 = resfin_test.join(error_df)

resfin2['abs_norm_error'] = abs(resfin2['normalized error'])

#now let's select only the errors that are 2 standard deviations away from the mean

resfin2['massive underestimation'] = resfin2['normalized error']<-2 
plt.figure(figsize=(12,8))

plt.scatter(resfin2['predprice'], resfin2['actprice'], c = resfin2['massive underestimation'])

plt.plot([0.2*10**6, 1.75*10**6], [0.2*10**6, 1.75*10**6], c = "red")

plt.legend(loc = "upper left")
#plot the residuals

plt.figure(figsize=(15,8))

sns.distplot(res.resid, fit=norm)
# Move our features into the X DataFrame

X = houses_o.loc[:,['sqft_living','condition', 'yr_built']]



# Move our labels into the y DataFrame

y = houses_o.loc[:,['price']] 
# separate y and X into train and test

X_train, X_test, y_train, y_test = train_test_split(

                                                    X, 

                                                    y, 

                                                    test_size=0.3, 

                                                    random_state=42

                                                   )
#train a basic multiple regression model and print out the coefficients

mod = sm.OLS(y_train, X_train)

res = mod.fit()

print(res.summary())
#plot the residuals

plt.figure(figsize=(15,8))

sns.distplot(res.resid, fit=norm)
#partial regression plots

fig = plt.figure(figsize=(12,8))

fig = sm.graphics.plot_partregress_grid(res, fig=fig)