# Code you have previously used to load data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor




# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# Create target object and call it y
y = home_data.SalePrice
# Create X
features = ['MSSubClass','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GarageArea','PoolArea']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)
data = home_data[['LotArea', 'YearBuilt','TotalBsmtSF', '1stFlrSF', '2ndFlrSF','GarageArea','PoolArea','MSSubClass','MSZoning','Condition1','OverallQual','OverallCond','FullBath','TotRmsAbvGrd','BsmtQual','HeatingQC','KitchenQual','SalePrice']]
test = test_data[['LotArea', 'YearBuilt','TotalBsmtSF', '1stFlrSF', '2ndFlrSF','GarageArea','PoolArea','MSSubClass','MSZoning','Condition1','OverallQual','OverallCond','FullBath','TotRmsAbvGrd','BsmtQual','HeatingQC','KitchenQual']]
data.head()
data.describe()
data.describe(include='O')
data = pd.get_dummies(data, columns = ['MSZoning','Condition1','OverallQual','OverallCond','BsmtQual','HeatingQC','KitchenQual'])
test = pd.get_dummies(test, columns = ['MSZoning','Condition1','OverallQual','OverallCond','BsmtQual','HeatingQC','KitchenQual'])
data.head()
data = data.drop(['MSZoning_RM','Condition1_RRNn','OverallQual_10','OverallCond_9','BsmtQual_TA','HeatingQC_TA','KitchenQual_TA'],axis=1)
test = test.drop(['MSZoning_RM','Condition1_RRNn','OverallQual_10','OverallCond_9','BsmtQual_TA','HeatingQC_TA','KitchenQual_TA'],axis=1)
data.head()
test.head()
y = data['SalePrice']
X = data.drop('SalePrice',axis=1)
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=65)
from xgboost import XGBRegressor
#from sklearn.model_selection import GridSearchCV

#parameters = {'max_depth':[5,10,15,20,25,30],'n_estimators':[50,100,150,200,250,300,400],
#             'learning_rate':[0.02,0.03,0.04,0.05,0.06,0.07],'random_state':[1]}

#def get_mae(num, train_X, val_X, train_y, val_y):
#    model = XGBRegressor(max_depth=25,n_estimators=num,learning_rate=0.04,random_state=1)
#    model.fit(train_X,train_y,early_stopping_rounds=5,eval_set=[(val_X, val_y)],verbose=False)
#    pred = model.predict(val_X)
#    return mean_absolute_error(pred, val_y)

#for num in esti:
#    mae = get_mae(num, train_X, val_X, train_y, val_y)
#    print("esti = {} , mae = {}".format(num,mae))

#model = XGBRegressor()
#clf = GridSearchCV(model,parameters)
#clf.fit(train_X,train_y,early_stopping_rounds=5,eval_set=[(val_X, val_y)],verbose=False)

#pred = clf.predict(val_X)
#print(mean_absolute_error(pred, val_y))

#rf_model_on_full_data = XGBRegressor(max_depth=25,n_estimators=200,learning_rate=0.04,random_state=1)
#rf_model_on_full_data.fit(X,y,verbose=False)
#clf.best_estimator_
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=65)

model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.07, max_delta_step=0,
       max_depth=5, min_child_weight=1, missing=None, n_estimators=200,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=1,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)

model.fit(X,y,early_stopping_rounds=5,eval_set=[(val_X, val_y)],verbose=False)
# make predictions which we will submit. 
test_preds = model.predict(test)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission_5.csv', index=False)
