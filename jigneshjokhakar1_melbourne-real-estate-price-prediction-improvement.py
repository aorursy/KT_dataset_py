import pandas as pd

import numpy as np

import scipy as sp

import sklearn

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

import scipy, pylab

from math import radians, sin, cos, acos,atan2, sqrt

import math

warnings.filterwarnings('ignore')

import xgboost as xgb

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib



df=pd.read_csv('../input/Copy of _sold.csv')

# eliminating irrelavant parameters and cleaning the data

df=df.drop(columns=['postcode','region','streetAddress','title','Unnamed: 12','Unnamed: 13'])

df['price']=df['price'][df['price']!='Contact agent'].apply(lambda x:(float(x[1:len(x)].replace(',',''))))


plt.scatter(df['price'],df['longitude'])

plt.xlabel('price')

plt.ylabel('longitude')

plt.show()#f, ax = plt.subplots(figsize=(12, 9))



plt.scatter(df['price'],df['latitude'])

plt.xlabel('price')

plt.ylabel('latitude')

plt.show()#f, ax = plt.subplots(figsize=(12, 9))



corrmat = df.corr()

sns.heatmap(corrmat, annot=True,vmax=.8, square=True);

plt.show()
dff=df[(df['price']<4000000) & (df['longitude']<145.5) & (df['longitude']>144)& (df['latitude']<-37.5)&(df['propertyType']=='house')]

dff_sp=dff

plt.scatter(dff['price'],dff['longitude'])

plt.xlabel('price')

plt.ylabel('longitude')

plt.show()



plt.scatter(dff['price'],dff['latitude'])

plt.xlabel('price')

plt.ylabel('latitude')

plt.show()



corrmat = dff.corr()

sns.heatmap(corrmat, annot=True,vmax=.8, square=True);

plt.show()

dff=dff.drop(['suburb','propertyType'],axis=1)

dff = dff.fillna(dff.mean())

X_train = dff[:dff.shape[0]].drop('price',axis=1)

X_test = dff[dff.shape[1]:].drop('price',axis=1)

y = dff.price

#print(X_test.head())

#print(X_train.head())

def r2_cv(model):

    r2= (cross_val_score(model, X_train, y, scoring="r2", cv = 5))

    return(r2)

model_ridge = Ridge()



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [r2_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]





cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot()

plt.xlabel("alpha")

plt.ylabel("r2")

plt.show()

print("Ridge model R-squared value: "+str(cv_ridge.mean()))





model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 0.00005]).fit(X_train, y)



print("Lasso model R-squared value: "+str(r2_cv(model_lasso).mean()))

dff=dff_sp.drop(['suburb','propertyType'],axis=1)

dff = dff.fillna(dff.mean())

dff['price']=np.log(dff['price'])

X_train = dff[:dff.shape[0]].drop('price',axis=1)

X_test = dff[dff.shape[1]:].drop('price',axis=1)

y = dff.price

#print(X_test.head())

#print(X_train.head())

def r2_cv(model):

    r2= (cross_val_score(model, X_train, y, scoring="r2", cv = 5))

    return(r2)

model_ridge = Ridge()



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [r2_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]





cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot()

plt.xlabel("alpha")

plt.ylabel("R-square")

plt.show()

print("Ridge model R-squared value for Log-price: "+str(cv_ridge.mean()))





model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 0.00005]).fit(X_train, y)



print("Lasso model R-squared value for Log-price: "+str(r2_cv(model_lasso).mean()))
lon1=df[df['suburb']=='Southbank']['longitude'].values[0]

lat1=df[df['suburb']=='Southbank']['latitude'].values[0]





d_map=df[(df['longitude']>df[df['suburb']=='Southbank']['longitude'].values[0])].dropna()

R = 6373.0

distance=[]



lat1=df[df['suburb']=='Southbank']['latitude'].values[0]

lon1=df[df['suburb']=='Southbank']['longitude'].values[0]

d_map['dlon'] = (df['longitude'] - lon1)/2#df[df['suburb']=='Southbank']['longitude'].values[0]

d_map['dlat'] = (df['latitude'] - lat1)/2#

#dlat=dlat/2



d_map['a']=np.power(np.sin(np.radians(d_map['dlat'])),2)+np.cos(np.radians(lat1))*np.cos(np.radians(df['latitude']))*np.power(np.sin(np.radians(d_map['dlon'])),2)

d_map['c']=2*np.arctan2(np.sqrt(d_map['a']),np.sqrt(1-d_map['a']))

d_map['distance']=R*d_map['c']



plt.scatter(d_map['distance'],d_map['price'])

plt.xlabel('distance')

plt.ylabel('price')

plt.title('Original price vs. distance')

plt.show()



d_map['price']=np.log(d_map['price'])



plt.scatter(d_map['distance'],d_map['price'])

plt.xlabel('distance')

plt.ylabel('price')

plt.title('log(price) vs. distance')

plt.show()



d_map['bearing']=np.degrees(np.arctan2(np.cos(np.radians(df['latitude']))*np.sin(np.radians(2*d_map['dlon'])),np.cos(np.radians(lat1))*np.sin(np.radians(df['latitude']))-np.sin(np.radians(lat1))*np.cos(np.radians(df['latitude']))*np.cos(np.radians(2*d_map['dlon']))))

d_map=d_map[(d_map['distance']<80) & (d_map['price']<15)].drop(['a','c','dlon','dlat'],axis=1)

d_map=d_map.drop(['propertyType','suburb'],axis=1)

d_map = d_map.fillna(d_map.mean())

X_train = d_map[:d_map.shape[0]].drop('price',axis=1)

X_test = d_map[d_map.shape[1]:].drop('price',axis=1)

y = d_map.price



def r2_cv(model):

    rmse= (cross_val_score(model, X_train, y, scoring="r2", cv = 5))

    return(rmse)

model_ridge = Ridge()



alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [r2_cv(Ridge(alpha = alpha)).mean() for alpha in alphas]





cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot()

plt.xlabel("alpha")

plt.ylabel("R-square")

plt.show()

print("Ridge model R-squared value for Log-price: "+str(cv_ridge.mean()))



model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)



print("Lasso model R-squared value for Log-price: "+str(r2_cv(model_lasso).mean()))
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values()])

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")

plt.show()