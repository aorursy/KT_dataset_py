import pandas as pd

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
path = '../input/california-housing-prices/housing.csv'

my_data = pd.read_csv(path)

my_data.head(10)

my_data.columns
y=my_data['median_house_value']

features=[ 'total_rooms', 'latitude', 'population', 'households', 'median_income']

X = my_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model=DecisionTreeRegressor(random_state=1)

my_model.fit(train_X, train_y)

pred=my_model.predict(val_X)

my_mae=mean_absolute_error(pred, val_y)

print('Result: {:,.0f}'.format(my_mae))



def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    my_model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    my_model.fit(train_X, train_y)

    pred=my_model.predict(val_X)

    my_mae=mean_absolute_error(pred, val_y)

    return my_mae
for i in [50, 100, 150, 200]:

    mae=get_mae(i,  train_X, val_X, train_y, val_y)

    print('%d  %d' %(i, mae))
score = {leaf: get_mae(leaf,  train_X, val_X, train_y, val_y) for leaf in [50, 100, 150, 200]}

best_value=min(score, key=score.get)

print(best_value)

my_model=DecisionTreeRegressor(max_leaf_nodes=best_value, random_state=1)

my_model.fit(X, y)

pred=my_model.predict(val_X)

my_mae=mean_absolute_error(pred, val_y)

print('Result: {:,.0f}'.format(my_mae))
my2_model=RandomForestRegressor(random_state=1)

my2_model.fit(train_X, train_y)

pred=my2_model.predict(val_X)

my_mae=mean_absolute_error(pred, val_y)

print('Result: {:,.0f}'.format(my_mae))