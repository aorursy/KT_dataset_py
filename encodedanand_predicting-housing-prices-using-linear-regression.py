import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
housing = pd.read_csv('../input/housing.csv')
housing.head()
housing.info()
housing[housing['total_bedrooms'].isnull()]
housing.loc[290]
housing.hist(bins=50, figsize=(20,20))
plt.show()
housing['total_bedrooms'][housing['total_bedrooms'].isnull()] = np.mean(housing['total_bedrooms'])
housing.loc[290]
housing['avg_rooms'] = housing['total_rooms']/housing['households']
housing['avg_bedrooms'] = housing['total_bedrooms']/housing['households']
housing.head()
housing.corr()
housing['pop_household'] = housing['population']/housing['households']
housing[:10]
housing['NEAR BAY']=0
housing['INLAND']=0
housing['<1H OCEAN']=0
housing['ISLAND']=0
housing['NEAR OCEAN']=0
housing.head()
housing.loc[housing['ocean_proximity']=='NEAR BAY','NEAR BAY']=1
housing.loc[housing['ocean_proximity']=='INLAND','INLAND']=1
housing.loc[housing['ocean_proximity']=='<1H OCEAN','<1H OCEAN']=1
housing.loc[housing['ocean_proximity']=='ISLAND','ISLAND']=1
housing.loc[housing['ocean_proximity']=='NEAR OCEAN','NEAR OCEAN']=1
housing.head()
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split


train_x = housing.drop(['total_rooms','total_bedrooms','households',
                        'ocean_proximity','median_house_value'],axis=1)
train_y = housing['median_house_value']

X,test_x,Y,test_y = train_test_split(train_x, train_y, test_size=0.2)
clf = LinearRegression()
clf.fit(np.array(X),Y)
import math

def roundup(x):
   return int(math.ceil(x / 100.0)) * 100 
pred = list(map(roundup,clf.predict(test_x)))

print(pred[:10])
test_y[:10]
from sklearn.metrics import mean_squared_error

predictions = clf.predict(test_x)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)
rmse