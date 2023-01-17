# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import datetime 

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from matplotlib import rcParams

#rcParams.update({'figure.autolayout': True})





#Load csv file to pandas

#train = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

train = pd.read_csv("../input/kc_house_data.csv")





#Quick look at data and summary statistics

train.head()

train.describe()
train.isnull().any()

train.dtypes
#Living Area and Price XY Scatter

var = 'sqft_living'

data = pd.concat([train['price'], train[var]], axis=1)

plot1=data.plot.scatter(x=var, y='price' )

plot1.axes.set_title('Price and SqFt Living Area')

plot1.set_xlabel("Square Ft Living Area")

plot1.set_ylabel("Price")

sns.plt.show()

#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 12))

plot2 =sns.heatmap(corrmat, vmax=.8);

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plot2.axes.set_title('Correlation Heat Map')

sns.plt.show()
#price correlation matrix

cmap1 = sns.cubehelix_palette(as_cmap=True)

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'price')['price'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True,cmap=cmap1, square=True, annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

hm.axes.set_title('Correlation Matrix')

plt.xticks(rotation=90)

plt.yticks(rotation=45)

plt.show()
#histogram and normal probability plot

#dist = sns.distplot(train['price'], fit = norm)

#dist.axes.set_title('Home Price Dist vs. Normal Dist')

#dist.set_xlabel("Home Price")

#sns.plt.show()

##applying log transformation

train['log_price'] = np.log(train['price'])



#Re-examine log_price distribution

#dist=sns.distplot(train['log_price'], fit=norm)

#dist.axes.set_title('Home Price Dist vs. Normal Dist')

#dist.set_xlabel("Home Price")

#sns.plt.show()
train.head()
#trying some feature egineering and sacling



train['age'] = 2017 -  train.yr_built

train.head()



train['sqft_feat'] = ((train.sqft_living - train.sqft_living.mean())/

                      (train.sqft_living.max() - train.sqft_living.min()))

train['bedroom_feat'] = ((train.bedrooms - train.bedrooms.mean())/

                      (train.bedrooms.max() - train.bedrooms.min()))

train['bath_feat'] = ((train.bathrooms - train.bathrooms.mean())/

                      (train.bathrooms.max() - train.bathrooms.min()))

train['sqft_lot_feat'] = ((train.sqft_lot - train.sqft_lot.mean())/

                      (train.sqft_lot.max() - train.sqft_lot.min()))



train = train.drop(['sqft_living','bedrooms', 'bathrooms', 'sqft_lot'], axis=1)



train.head()

#Massaging Data





#Create Dummy variable(0,1) for renovated

train['renovated']=0

train.loc[train['yr_renovated'] > 0, 'renovated'] = 1

train = train.drop(['yr_renovated'], axis=1)



#has basement

train['has_basement']=0

train.loc[train['sqft_basement']>0, 'has_basement']=1

train = train.drop(['sqft_basement'], axis=1)



#Drop non needed columns 

train = train.drop(['zipcode', 'lat','long','sqft_living15','sqft_above','sqft_lot15','id','date','price','yr_built'], axis=1)

train = train.drop(['view','condition', 'grade'], axis=1)

train.head()

train.describe()




#Quick Regression model and look at coefficients

x_train = train.drop("log_price", axis=1)

y_train = train['log_price']

lr = LinearRegression()

lr.fit(x_train, y_train)



#Plot Coefficients

coefs = pd.Series(lr.coef_, index = x_train.columns)

coefs.plot(kind = "barh")

plt.title("Coefficients in the Linear Regression Model")

plt.show()



#Clearly not the most accurate model....


#Ridge

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3])

ridge.fit(x_train, y_train)

alpha = ridge.alpha_

#Plot Ridgecoefs2 = pd.Series(ridge.coef_, index = x_train.columns)

coefs2 = pd.Series(ridge.coef_, index = x_train.columns)

coefs2.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()