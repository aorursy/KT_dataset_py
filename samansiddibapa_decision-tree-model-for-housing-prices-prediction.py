import pandas as pd



#reading the data into a DataFrame

home_data = pd.read_csv('../input/home-data-for-ml-course/train.csv')
#setting the display to show all rows and columns stat

pd.set_option('display.max_rows',None,'display.max_columns',None)



#displaying descriptive stats

home_data.describe(include='all')
home_data.head()
#selecting prediction target

y = home_data.SalePrice



#selecting input features

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)
from sklearn.tree import DecisionTreeRegressor



iowa_model = DecisionTreeRegressor(random_state=1)



iowa_model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error



val_predictions = iowa_model.predict(val_X)



val_mae = mean_absolute_error(val_predictions, val_y)



print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)
candidate_max_leaf_nodes = [5*n for n in range(1,100)]



maes=[]

for max_leaf_nodes in candidate_max_leaf_nodes:

    mae= get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    maes.append(mae)

    if mae == min(maes):

        best_tree_size=max_leaf_nodes
best_iowa_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

best_iowa_model.fit(train_X, train_y)

val_predictions = best_iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
test_data_path = '../input/home-data-for-ml-course/test.csv'



test_data = pd.read_csv(test_data_path)



features=['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

test_X = test_data[features]

 

test_preds = best_iowa_model.predict(test_X)
output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('decision_tree_output.csv', index=False)