import pandas as pd
from sklearn.tree import DecisionTreeRegressor
file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(file_path)
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X= home_data[features]
y
X

model = DecisionTreeRegressor()
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
model.fit(train_X, train_y)
predictions = model.predict(val_X)
predictions
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(val_y,predictions))