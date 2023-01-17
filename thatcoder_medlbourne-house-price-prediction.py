import numpy as np # linear algebra

import pandas as pd # data processing



melbourne_file_path = '../input/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path); 

melbourne_data.describe()
melbourne_data.columns
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

x.describe()
x.head()
from sklearn.tree import DecisionTreeRegressor



melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(x, y)
print("Making predictions for the following 5 houses:")

print(x.head())

print("The predictions are")

print(melbourne_model.predict(x.head()))