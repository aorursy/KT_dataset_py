import pandas as pd
# save filepath
houses_file_path = "../input/train.csv"
# read the data and store data in DataFrame 
houses_data = pd.read_csv(houses_file_path)
# fill nall by mean()
houses_data = houses_data.fillna(houses_data.mean())
houses_data.describe()
# Choose target
y = houses_data.SalePrice
# Choose predictors
houses_predictors = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
X = houses_data[houses_predictors]

from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Use  Random Forest
forest_model = RandomForestRegressor()
# fit the data
forest_model.fit(train_X, train_y)
# Use the model to make predictions
house_preds = forest_model.predict(val_X)
print(house_preds)
# print(mean_absolute_error(val_y, house_preds))
my_submission = pd.DataFrame({'Id': houses_data.mean().Id, 'SalePrice': house_preds})
my_submission.to_csv('submissionHousePrices.csv', index=False)
