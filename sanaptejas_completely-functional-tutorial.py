import pandas as pd

from sklearn.tree import DecisionTreeRegressor



melb = pd.read_csv('../input/melb_data.csv')

melb.info()
melb.columns
predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Price']
clean_set = melb[predictors].dropna()
clean_set.head()
y = clean_set.Price
del clean_set['Price']
X = clean_set
model = DecisionTreeRegressor()
model.fit(X,y)
print("prediction for 5 houses:")

print(X.head())



print("predictions are: ")

print(model.predict(X.head()))
from sklearn.metrics import mean_absolute_error
prediction = model.predict(X)     
mean_absolute_error(y, prediction)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X,y, random_state=0)
model.fit(X_train, Y_train)
predictions2 = model.predict(X_val)
mean_absolute_error(Y_val, predictions2)
for max_leaf_nodes in [5,50,500,5000]:

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(X_train, Y_train)

    print("{} nodes, MAE is {}".format(max_leaf_nodes, mean_absolute_error(Y_val, model.predict(X_val))))