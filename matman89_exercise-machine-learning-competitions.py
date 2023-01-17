### Kaggle Competition 1 -- House Prediction Data ###



import pandas as pd

import matplotlib.pyplot as plt

import warnings

import numpy

from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from pandas.plotting import scatter_matrix

warnings.filterwarnings('ignore')



##Load Data Set

path = '../input/train.csv'

Testpath = '../input/test.csv'

house_data = pd.read_csv(path)

test_data = pd.read_csv(Testpath)

my_imputer = SimpleImputer()



##Setting X (features) and Y (prediction) variables

features = ['LotArea', 'YearBuilt', '1stFlrSF',

            'TotalBsmtSF', 'OverallQual', 'GarageArea', 

             '2ndFlrSF', 'TotRmsAbvGrd', 'BsmtFinSF1','Fireplaces']



categorical = ['BldgType','Neighborhood','FireplaceQu',

               'BsmtFinType1','GarageType','SaleCondition']



categ_filt = [cname for cname in house_data[categorical] if 

                                house_data[cname].nunique() < 35]



features = features+categ_filt

x = pd.get_dummies(house_data[features])

test_x = pd.get_dummies(test_data[features])

imputed_test_x = my_imputer.fit_transform(test_x)

test_x_imp = pd.DataFrame(imputed_test_x, columns = test_x.columns)



##Will need to impute: TotalBasmtSF

y = house_data.SalePrice



##Need to split the data into validation and training data

##This way you can more accurately see how the model will

##perform against data it hasn't seen before



train_x, val_x, train_y, val_y = train_test_split(x,y,

                                                  random_state = 1)

##Create your models and test each



##Decision Tree

model = DecisionTreeRegressor(random_state = 1)

model.fit(train_x, train_y)

preds = model.predict(val_x)

mae = mean_absolute_error(preds, val_y)

print("MAE single decision tree: {:,.0f}".format(mae))



##Decision Tree Max Node

model = DecisionTreeRegressor(max_leaf_nodes=100, random_state = 1)

model.fit(train_x, train_y)

preds = model.predict(val_x)

mae = mean_absolute_error(preds, val_y)

print("MAE with Max Nodes: {:,.0f}".format(mae))



##Random Forest Model

model = RandomForestRegressor(random_state=1)

model.fit(train_x,train_y)

preds = model.predict(val_x)

mae = mean_absolute_error(preds,val_y)

print('MAE with Random Forest: {:,.0f}'.format(mae))



##Random Forest Model Max Node

model = RandomForestRegressor(max_leaf_nodes=420, random_state=1)

model.fit(train_x,train_y)

preds = model.predict(val_x)

mae = mean_absolute_error(preds,val_y)

print('MAE with Random Forest Max Node: {:,.0f}'.format(mae))



## XGB Model Max Node

model = XGBRegressor(learning_rate=0.1,n_estimators=300)

model.fit(train_x,train_y,verbose=False)

preds = model.predict(val_x)

mae = mean_absolute_error(preds,val_y)

print('MAE with XGB: {:,.0f}'.format(mae))







# To improve accuracy, create a new model which you will train on all training data

xgb_model_on_full_data = XGBRegressor(learning_rate=0.1,n_estimators=300)



# fit model on all data from the training data

xgb_model_on_full_data.fit(x, y)

test_preds = xgb_model_on_full_data.predict(test_x_imp)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('submissionXGB.csv', index=False)


