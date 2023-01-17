import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
house = pd.read_csv("/Users/qiaohan/Desktop/kc_house_data.csv")
house.head()
#Print col names
for col in house.columns: 
    print(col)
#checking for categorical data
print(house.dtypes)
#checking for nulls
print(house.isnull().any())
#dropping id & date columns
dat = house.drop(['id','date'], axis = 1)
dat.head()
features = ['price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=20)
sns.heatmap(dat[features].corr(),annot=True)
#setting X & Y
#x = dat.drop(['sqft_living'], axis = 1).values
#y = dat.iloc[:,3].values
x = dat.drop(['sqft_living'], axis = 1)
y = dat.iloc[:,3]
#split train & test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#x_train
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.coef_)
# predict using the testing data
y_pred = regressor.predict(x_test)
print(y_pred)

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print('Mean squared error:',mean_squared_error(y_test, y_pred))
# The coefficient of determination(R^2)
print('Coefficient of determination:',r2_score(y_test, y_pred))