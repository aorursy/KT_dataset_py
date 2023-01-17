# Code you have previously used to load data

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor





# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



home_data.head()
model=RandomForestRegressor(n_estimators=100,random_state=1)

model.fit(train_X,train_y)
predict_y=model.predict(val_X)
mean_absolute_error(val_y,predict_y)