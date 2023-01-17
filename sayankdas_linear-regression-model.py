import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pandas.tools.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
#Read data
house = pd.read_csv('../input/housing.csv')
house.head()
#Size of the datahouse.shape
house.corr()
corr = house.corr()
corr.sort_values(["median_house_value"], ascending = False, inplace = True)
print(corr.median_house_value)
house['median_income-s2'] = house['median_income']**2
house['median_income-s3'] = house['median_income']**3
house['median_income-sq'] = np.sqrt(house['median_income'])


#How house data looking
house.info()
#Check for missing values
house.isnull().any()
#Missing data in total_bedrooms
#Handling missing values
median = house['total_bedrooms'].median()
house['total_bedrooms'].fillna(median,inplace=True)
house.isnull().any()
house_cat = house['ocean_proximity']
house_cat.head(10)
house_cat.value_counts()
y = house['median_house_value']
house = house.drop(['median_house_value'],axis=1)

house.head()
cat_data = house['ocean_proximity']
house = house.drop(['ocean_proximity'],axis = 1)
cat_data = pd.get_dummies(cat_data)
cat_data.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(house)
housing_data = scaler.transform(house)
full_data = np.append(housing_data, cat_data, axis = 1)
print(full_data.shape)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
lr = LinearRegression()
lr.fit(house, y)
lr.score(house,y)
house_prediction = lr.predict(house[0:5])
house_prediction[0:5]
y[0:5]
from sklearn.metrics import mean_squared_error

housing_predictions = lr.predict(house)
lin_mse = mean_squared_error(y, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
