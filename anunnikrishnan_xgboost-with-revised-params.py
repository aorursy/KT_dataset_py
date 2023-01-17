# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,mean_absolute_error

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()  # for plot styling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV





#######################



## load pima indians dataset

train = pd.read_csv("../input/train.csv")

test =  pd.read_csv("../input/test.csv")







score = {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1,'NA': 0}

score_garage = {'2Types': 6,'Attchd': 5, 'Basment': 4,'BuiltIn': 3, 'CarPort': 2,'Detchd': 1, 'NA': 0}

score_shape = {'Reg': 4,'IR1': 3, 'IR2': 2,'IR3':1}



train['LotShape_'] = train['LotShape'].copy()

test['LotShape_'] = test['LotShape'].copy()



train['LotShape'] = train.replace({"LotShape": score_garage})

test['LotShape'] = test.replace({"LotShape": score_garage})

train['LotShape'] = train['LotShape'].astype(float)

test['LotShape'] = test['LotShape'].astype(float)





train['GarageType_'] = train['GarageType'].copy()

test['GarageType_'] = test['GarageType'].copy()

train = train.replace({"GarageType": score_garage})

test= test.replace({"GarageType": score_garage})

train['GarageType'] = train['GarageType'].astype(float)

test['GarageType'] = test['GarageType'].astype(float)



train['GarageQual_'] = train['GarageQual'].copy()

test['GarageQual_'] = test['GarageQual'].copy()

train = train.replace({"GarageQual": score})

test= test.replace({"GarageQual": score})

train['GarageQual'] = train['GarageQual'].astype(float)

test['GarageQual'] = test['GarageQual'].astype(float)



scr = {'Fin': 4,'RFn': 3, 'Unf': 2,'NA': 1}



train = train.replace({"GarageFinish": scr})

test = test.replace({"GarageFinish": scr})

train['GarageFinish'] = train['GarageFinish'].astype(float)

test['GarageFinish'] = test['GarageFinish'].astype(float)



train['KitchenQual_'] = train['KitchenQual'].copy()

test['KitchenQual_'] = test['KitchenQual'].copy()

train['KitchenQual']  = train.replace({"KitchenQual": score})

test['KitchenQual'] = test.replace({"KitchenQual": score})

train['KitchenQual'] = train['KitchenQual'].astype(float)

test['KitchenQual'] = test['KitchenQual'].astype(float)



train['GarageQual_Area_Type'] = train['GarageFinish'] * train['GarageQual'] * train['GarageArea'] * train['GarageType']

test['GarageQual_Area_Type'] = test['GarageFinish'] * test['GarageQual'] * test['GarageArea'] * test['GarageType']



train['Garage_Area_Type'] =  train['GarageArea'] * train['GarageType'] 

test['Garage_Area_Type'] =  test['GarageArea'] * test['GarageType']



train['GarageQual_Type'] = train['GarageQual']  * train['GarageType']

test['GarageQual_Type'] =  test['GarageQual']  * test['GarageType']



train['GarageQual_Area'] = train['GarageQual']  * train['GarageArea']

test['GarageQual_Area'] =  test['GarageQual']  * test['GarageArea']





### multi tri



train['Garage_Area_Type3'] =  train['GarageArea'] * train['GarageType'] * train['GarageFinish']

test['Garage_Area_Type3'] =  test['GarageArea'] * test['GarageType'] * test['GarageFinish']



train['GarageQual_Type3'] = train['GarageQual']  * train['GarageType'] * train['GarageFinish']

test['GarageQual_Type3'] =  test['GarageQual']  * test['GarageType'] * test['GarageFinish']



train['GarageQual_Area3'] = train['GarageQual']  * train['GarageArea'] * train['GarageFinish']

test['GarageQual_Area3'] =  test['GarageQual']  * test['GarageArea'] * test['GarageFinish']







train['kitchenQual_Area'] = train['KitchenQual'] * train['KitchenAbvGr']

test['kitchenQual_Area'] = test['KitchenQual'] * test['KitchenAbvGr']



train['FireplaceQu_'] = train['FireplaceQu'].copy()

test['FireplaceQu_'] = test['FireplaceQu'].copy()



train = train.replace({"FireplaceQu": score})

test = test.replace({"FireplaceQu": score})

train['FireplaceQu'] = train['FireplaceQu'].astype(float)

test['FireplaceQu'] = test['FireplaceQu'].astype(float)





train['FireplaceQu_Number'] = train['FireplaceQu'] * train['Fireplaces']

test['FireplaceQu_Number'] = test['FireplaceQu'] * test['Fireplaces']



train['PoolQC_'] = train['PoolQC'].copy()

test['PoolQC_'] = test['PoolQC'].copy()



train = train.replace({"PoolQC": score})

test = test.replace({"PoolQC": score})

train['PoolQC'] = train['PoolQC'].astype(float)

test['PoolQC'] = test['PoolQC'].astype(float)





train['PoolQC_Area'] = train['PoolQC'] * train['PoolArea']

test['PoolQC_Area'] = test['PoolQC'] * test['PoolArea']



train['HeatingQC_'] = train['HeatingQC'].copy()

test['HeatingQC_'] = test['HeatingQC'].copy()

train = train.replace({"HeatingQC": score})

test = test.replace({"HeatingQC": score})

train['HeatingQC'] = train['HeatingQC'].astype(float)

test['HeatingQC'] = test['HeatingQC'].astype(float)





train['PoolQC_Area'] = train['PoolQC'] * train['PoolArea']

test['PoolQC_Area'] = test['PoolQC'] * test['PoolArea']



train['LotShape_no_Area'] = train['LotShape'] * train['LotArea']

test['LotShape_no_Area'] = test['LotShape'] * test['LotArea']





"""

cost_score = {'FV': 5,'RM': 4, 'RH': 3,'RL': 2, 'C (all)': 1,'I': 0}

train['MSZoning'] = train.replace({"MSZoning": score})

test['MSZoning'] = test.replace({"MSZoning": score})

train['MSZoning'] = train['MSZoning'].astype(float)

test['MSZoning'] = test['MSZoning'].astype(float)

"""



train['Overall_state'] = train['OverallQual'] * train['OverallCond']

test['Overall_state'] = test['OverallQual'] * test['OverallCond']



base_score = {'Ex': 5,'Gd': 4, 'TA': 3,'Fa': 2, 'Po': 1,'NA': 0}



train['BsmtQual_'] = train['BsmtQual'].copy()

test['BsmtQual_'] = test['BsmtQual'].copy()

train['BsmtQual'] = train.replace({"BsmtQual": base_score})

test['BsmtQual'] = test.replace({"BsmtQual": base_score})

train['BsmtQual'] = train['BsmtQual'].astype(float)

test['BsmtQual'] = test['BsmtQual'].astype(float)



train['BsmtCond_'] = train['BsmtCond'].copy()

test['BsmtCond_'] = test['BsmtCond'].copy()

train['BsmtCond'] = train.replace({"BsmtCond": base_score})

test['BsmtCond'] = test.replace({"BsmtCond": base_score})

train['BsmtCond'] = train['BsmtCond'].astype(float)

test['BsmtCond'] = test['BsmtCond'].astype(float)



base_score = {'Gd': 4, 'AV': 3,'Mn': 2, 'No': 1,'NA': 0}



train['BsmtExposure_'] = train['BsmtExposure'].copy()

test['BsmtExposure_'] = test['BsmtExposure'].copy()

train['BsmtExposure'] = train.replace({"BsmtExposure": base_score})

test['BsmtExposure'] = test.replace({"BsmtExposure": base_score})

train['BsmtExposure'] = train['BsmtExposure'].astype(float)

test['BsmtExposure'] = test['BsmtExposure'].astype(float)



train['Abovegr_lotArea'] = train['LotArea'] / train['GrLivArea']

test['Abovegr_lotArea'] = test['LotArea'] / test['GrLivArea']





id = test['Id']

test.drop('Id', axis = 1,inplace  = True)

train.drop('Id', axis = 1,inplace  = True)



y = train['SalePrice']



X = train.drop(['SalePrice'], axis=1)



colm = X.select_dtypes(exclude=['int','float'])

col = colm.columns









# Create dummy variables for each level of `col`

train_animal_dummies = pd.get_dummies(X[col], prefix=col)

X = X.join(train_animal_dummies)



test_animal_dummies = pd.get_dummies(test[col], prefix=col)

test = test.join(test_animal_dummies)



# Find the difference in columns between the two datasets

# This will work in trivial case, but if you want to limit to just one feature

# use this: f = lambda c: col in c; feature_difference = set(filter(f, train)) - set(filter(f, test))

feature_difference = set(X) - set(test)



# create zero-filled matrix where the rows are equal to the number

# of row in `test` and columns equal the number of categories missing (i.e. set difference 

# between relevant `train` and `test` columns

feature_difference_df = pd.DataFrame(data=np.zeros((test.shape[0], len(feature_difference))),

                                     columns=list(feature_difference))



# add "missing" features back to `test

test = test.join(feature_difference_df)



test  = test.select_dtypes(exclude=['object'])

X  = X.select_dtypes(exclude=['object'])



c = X.columns

test = test[c]











# Initialize XGB and GridSearch

#model = xgb.XGBRegressor(n_estimators = 500,learning_rate=0.01)



X_train, X_test, y_train, y_test = train_test_split(

     X, y, test_size=0.35, random_state=42)



testX = X_test 

testY = y_test



# A parameter grid for XGBoost

paramGrid = {'learning_rate': [0.1,0.01],'n_estimators': [5000]}

"""

fit_params={"early_stopping_rounds":10, 

            "eval_metric" : "mae", 

            "eval_set" : [[testX, testY]]}

"""



# cv = None, default 3 fold cross validation







model = xgb.XGBRegressor()

gridsearch = RandomizedSearchCV(model, paramGrid, verbose=1 , cv=13)

gridsearch.fit(X_train,y_train)







# predict

y_pred = gridsearch.best_estimator_.predict(X_test)

predictions = [round(value) for value in y_pred]

# evaluate predictions

output_eval = pd.DataFrame(list(zip(y_test, predictions)),columns = ['y_test', 'predictions'])

print(output_eval)

print("mean abs error testing",mean_absolute_error(y_test, predictions))



y_pred_train = gridsearch.best_estimator_.predict(X_train)

predictions = [round(value) for value in y_pred_train]

# evaluate predictions

print("mean abs error train",mean_absolute_error(y_train, predictions))



#y_pred = grid.best_estimator_.predict(X_test)

#y_pred_train = grid.best_estimator_.predict(X_train)



"""

from sklearn.metrics import mean_squared_log_error,mean_squared_error

rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))

print("msle is ==== ",rmsle)

print("mse test",mean_squared_error(y_test, y_pred))

print("mse train",mean_squared_error(y_train, y_pred_train))

"""



#### final prediction #################



final_outcome = gridsearch.best_estimator_.predict(test)



final_outcome = pd.DataFrame(final_outcome)

final_outcome.columns = ['SalePrice']



id = pd.DataFrame(id)

id = id.reset_index()

id = id.drop('index',axis =1)



final_outcome = pd.concat([id,final_outcome],axis = 1)

final_outcome.to_csv("submission.csv",index = False)

print(final_outcome)