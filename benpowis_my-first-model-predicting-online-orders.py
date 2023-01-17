import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('../input/kaggle_train.csv', header=0).fillna(value=0, axis=1)

print(data.columns)

# print(data.describe())

print(data.info())

#data.fillna(value=0, axis=1)
y = data.orders

x = pd.get_dummies(data, columns=['device', 'user_type']).select_dtypes(exclude=['object','bool'])
x['revenue']=x['revenue'].fillna(0)

# print(x.describe())

print(x.head())
train_x, val_x, train_y, val_y = train_test_split(x, y,random_state = 0)

# Define model
model = RandomForestRegressor()

# Fit model
model.fit(train_x, train_y)
# get predicted orders
val_predictions = model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))
# The sum of a days orders?
print(model.predict(x).sum())
# what about a different data set?
data2 = pd.read_csv('../input/kaggle_test.csv', header=0)

x = pd.get_dummies(data2, columns=['device', 'user_type'])
x = x.select_dtypes(exclude=['object','bool'])
x['revenue']=x['revenue'].fillna(0)

print(model.predict(x).sum())
