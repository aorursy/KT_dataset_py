import pandas as pd

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score
housing_data=pd.read_csv('../input/melb_data.csv')

housing_data.columns
housing_data.describe()
housing_data=housing_data.dropna(axis=0)
housing_data.describe()
y=housing_data.Price
housing_features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
x=housing_data[housing_features]
x.describe()
x.head()
boxplot = x.boxplot(column=['Rooms', 'Bathroom', 'Landsize','Lattitude','Longtitude'])
scatter_matrix(x)

plt.show()
#Define the model

housing_model=DecisionTreeRegressor(random_state=1)
#Model fitting

housing_model.fit(x,y)
print("Making predictions now for top 5 observations.....")

print("The predictions are:")

predict_price=housing_model.predict(x.head())

predict_price_all=housing_model.predict(x)

print(predict_price)
print("Actual prices are:")

print(y.head())