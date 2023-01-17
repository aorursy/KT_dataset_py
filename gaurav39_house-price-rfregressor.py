import pandas as pd
data  = pd.read_csv('../input/train.csv')
#print (data)
print(data.columns)
saleprice =data.SalePrice
print(saleprice)
clm_of_interest = ['LotFrontage','LotArea']
data_of_two = data[clm_of_interest]
#data_of_two.describe()
data_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[data_predictors]
y = saleprice

from sklearn.tree import DecisionTreeRegressor
data_model  = DecisionTreeRegressor()
data_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The prediction are")
print(data_model.predict(X.head()))
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
data_model2 = DecisionTreeRegressor()
data_model2.fit(train_X, train_y)
val_prediction = data_model2.predict(val_X)
print(mean_absolute_error(val_y, val_prediction))


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
val_prediction2 = forest_model.predict(val_X)
print(mean_absolute_error(val_y, val_prediction2))
import numpy as np
import pandas as pd

#predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
new_model = RandomForestRegressor()
new_model.fit(X,y)
test = pd.read_csv('../input/test.csv')
test_X = test[data_predictors]
predicted_prices = new_model.predict(test_X)
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
