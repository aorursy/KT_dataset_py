import pandas as pd

import matplotlib.pyplot as plt
file_path = '../input/car-price-prediction/CarPrice_Assignment.csv'

price_prediction = pd.read_csv(file_path, index_col = 0)

price_prediction.isnull().sum()
price_prediction.describe()
price_prediction.head()
plt.figure(figsize = (15, 5))

price_prediction.price.plot(title = 'Variation of Car Statistics by Index')

price_prediction.peakrpm.plot(color = 'y')



plt.legend()

plt.show()
price_prediction.citympg.plot(figsize = (15, 5), title = 'Variation of Engine Statistics', color = 'g')

price_prediction.highwaympg.plot(color = 'r')

price_prediction.horsepower.plot(kind = 'bar', rot = 90)



plt.axes().xaxis.set_major_formatter(plt.NullFormatter())



plt.legend()

plt.show()
type_cars = price_prediction.groupby('CarName').size().sort_values(ascending = False)

type_cars.nlargest(15).plot(kind = 'bar', figsize = (20, 5), title = 'Most Popular Car Models', rot = 10)



plt.show()
price_prediction.dtypes
from sklearn.preprocessing import LabelEncoder

conversion = LabelEncoder()



price_prediction.CarName = conversion.fit_transform(price_prediction.CarName)

price_prediction.fueltype = conversion.fit_transform(price_prediction.fueltype)

price_prediction.aspiration = conversion.fit_transform(price_prediction.aspiration)

price_prediction.doornumber = conversion.fit_transform(price_prediction.doornumber)

price_prediction.carbody = conversion.fit_transform(price_prediction.carbody)

price_prediction.drivewheel = conversion.fit_transform(price_prediction.drivewheel)

price_prediction.enginelocation = conversion.fit_transform(price_prediction.enginelocation)

price_prediction.enginetype = conversion.fit_transform(price_prediction.enginetype)

price_prediction.cylindernumber = conversion.fit_transform(price_prediction.cylindernumber)

price_prediction.fuelsystem = conversion.fit_transform(price_prediction.fuelsystem)
price_prediction.dtypes
correlation = price_prediction.corr()

correlation.style.background_gradient(cmap = 'coolwarm', axis = None).set_precision(3)
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error
X = price_prediction.drop('price', axis = 1)

y = price_prediction.price



train_X, test_X, train_y, test_y = train_test_split(X, y)
from sklearn.tree import DecisionTreeRegressor



DTreeModel = DecisionTreeRegressor(random_state = 0)

DTreeModel.fit(train_X, train_y)



predval_DTree = DTreeModel.predict(test_X)

mean_error = mean_absolute_error(test_y, predval_DTree)

percent_error = (mean_error / price_prediction.price.mean()) * 100



print('Mean Prices in Original Model:', price_prediction.price.mean())

print('Mean Absolute Error of Predictions:', mean_error)

print('Percentage Error: {}%'.format(percent_error))
from sklearn.ensemble import RandomForestRegressor



RandForestModel = RandomForestRegressor(random_state = 1)

RandForestModel.fit(train_X, train_y)



predval_RandFor = RandForestModel.predict(test_X)

mean_error = mean_absolute_error(test_y, predval_RandFor)

percent_error = (mean_error / price_prediction.price.mean()) * 100



print('Mean Prices in Original Model:', price_prediction.price.mean())

print('Mean Absolute Error of Predictions:', mean_error)

print('Percentage Error: {}%'.format(percent_error))