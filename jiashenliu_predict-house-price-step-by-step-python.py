import pandas as pd

import numpy as np

df = pd.read_csv('../input/kc_house_data.csv')

print(df.shape)
df.head()
del df['id']
NA_Count = pd.DataFrame({'Sum of NA':df.isnull().sum()}).sort_values(by=['Sum of NA'],ascending=[0])

NA_Count['Percentage'] = NA_Count['Sum of NA']/df.shape[1]
sum(NA_Count['Percentage'])
from sklearn.model_selection import train_test_split

train,test = train_test_split(df,test_size = 0.2,random_state=42)
cat = ['waterfront','view','condition','grade']

con = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']
from ggplot import *

lonlat = ggplot(train,aes(x='long',y='lat',color='price'))+geom_point()+scale_color_gradient(low='white',high='red')+ggtitle('Color Map of Price') 

print(lonlat)
lonprice = ggplot(train,aes(x='long',y='price'))+geom_point()+ggtitle('Price VS Longitude')

print(lonprice)
def centralize_long(lon):

    return np.abs(lon+122.25)*-1
train['norm_lon'] = train['long'].apply(lambda x: centralize_long(x))

test['norm_lon'] = test['long'].apply(lambda x: centralize_long(x))
lonprice2 = ggplot(train,aes(x='norm_lon',y='price'))+geom_point()+ggtitle('Price VS Centered Longitude')

print(lonprice2)
latprice = ggplot(train,aes(x='lat',y='price'))+geom_point()+stat_smooth()+ggtitle('Price VS Latitude')

print(latprice)
zipprice = ggplot(train,aes(x='zipcode',y='price'))+geom_point()+ggtitle('ZipCode VS Price')

print(zipprice)
latlonzip = ggplot(train,aes(x='long',y='lat',color='zipcode'))+geom_point()+ggtitle('Long-Lat VS ZipCode')

print(latlonzip)
def zip2area(zipcode):

    if zipcode <= 98028:

        return 'A'

    elif zipcode>98028 and zipcode <= 98072:

        return 'B'

    elif zipcode>98072 and zipcode<98122:

        return 'C'

    else:

        return 'D'
train['Area'] = train['zipcode'].apply(lambda x:zip2area(x))

test['Area'] = test['zipcode'].apply(lambda x:zip2area(x))
con_train = train[con+['price']]

cor_tar_con = []

for each in con:

    cor_tar_con.append(np.corrcoef(train[each],train['price'])[0][1])

cor_label = pd.DataFrame({'Variables':con,'Correlation':cor_tar_con}).sort_values(by=['Correlation'],ascending=[0])
import matplotlib.pyplot as plt

from matplotlib import cm as cm

import seaborn as sns
pos_1 = np.arange(len(con))

plt.bar(pos_1, cor_label['Correlation'], align='center', alpha=0.5)

plt.xticks(pos_1, cor_label['Variables'],rotation='vertical')

plt.ylabel('Correlation')

plt.title('Correlation between price and variables') 

plt.show()
corr = con_train.corr()

corr.style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
def box_plot(var):

    pt = a = ggplot(train,aes(x=var,y='price'))+geom_boxplot() + theme_bw()+ggtitle('Boxplot of '+var+' and price')

    return print(pt)
for each in cat:

    box_plot(each)
train['date']=pd.to_datetime(train['date'])

test['date']=pd.to_datetime(test['date'])
dateprice = ggplot(train,aes(x='date',y='price'))+geom_line()+stat_smooth()+ggtitle('Date VS Price')

print(dateprice)
min_date = min(test['date'])

def get_interval(date):

    return int(str(date-min_date).split()[0])
train['date_interval'] = train['date'].apply(lambda x: get_interval(x))

test['date_interval']=test['date'].apply(lambda x: get_interval(x))
columns = con + cat + ['date_interval','norm_lon','Area']

train_ = train[columns]

test_ = test[columns]

train_['Area']=pd.factorize(train_['Area'], sort=True)[0]

test_['Area']=pd.factorize(test_['Area'], sort=True)[0]
import statsmodels.api as sm

fig=sm.qqplot(train['price'])

plt.show()
fig=sm.qqplot(np.log(train['price']))

plt.show()
train_['log_price'] = np.log(train['price'])
Models = []

RMSE = []
from math import sqrt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as mse
Models.append('Normal Linear Regression')

reg = LinearRegression(n_jobs=-1)

reg.fit(train_[columns],train_['log_price'])

pred = np.exp(reg.predict(test_))

Accuracy = sqrt(mse(pred,test['price']))

print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)

RMSE.append(Accuracy)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

pipe = Pipeline([

('sc',StandardScaler()),

('poly',PolynomialFeatures(include_bias=True)),

('reg',LinearRegression())

])

model = GridSearchCV(pipe,param_grid={'poly__degree':[2,3]})

model.fit(train_[columns],train_['log_price'])

degree = model.best_params_

print(degree)

pred = np.exp(model.predict(test_))

Accuracy = sqrt(mse(pred,test['price']))

print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)

RMSE.append(Accuracy)
Models.append('LinearRegression Step2 Polynominal')
from sklearn.linear_model import Lasso
pipe = Pipeline([

('sc',StandardScaler()),

('poly',PolynomialFeatures(degree=2,include_bias=True)),

('las',Lasso())

])

model = GridSearchCV(pipe,param_grid={'las__alpha':[0.0005,0.001,0.01]})

model.fit(train_[columns],train_['log_price'])

degree = model.best_params_

print(degree)

pred = np.exp(model.predict(test_))

Accuracy = sqrt(mse(pred,test['price']))

print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)

RMSE.append(Accuracy)

Models.append('Lasso')
from sklearn.linear_model import ElasticNet

pipe = Pipeline([

('sc',StandardScaler()),

('poly',PolynomialFeatures(degree=2,include_bias=True)),

('en',ElasticNet())

])

model = GridSearchCV(pipe,param_grid={'en__alpha':[0.005,0.01,0.05,0.1],'en__l1_ratio':[0.1,0.4,0.8]})

model.fit(train_[columns],train_['log_price'])

degree = model.best_params_

print(degree)

pred = np.exp(model.predict(test_))

Accuracy = sqrt(mse(pred,test['price']))

print('=='*20+'RMSE: '+str(Accuracy)+'=='*20)

RMSE.append(Accuracy)

Models.append('ElasticNet Regression')
RegSummary = pd.DataFrame({'Model':Models,'RMSE':RMSE})

summary = ggplot(RegSummary,aes(x='Model',weight='RMSE'))+geom_bar()+theme_bw()+ggtitle('Summary of Regression Model')

print(summary)
