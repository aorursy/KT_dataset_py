from pandas import read_csv

from xgboost import XGBRegressor

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score, mean_absolute_error

import numpy 

from sklearn.grid_search import GridSearchCV

# load data

data = read_csv('../input/train.csv')

data.head()
# Binarizing Categorical Variables

import pandas as pd

features = data.columns

cats = [feat for feat in features if 'cat' in feat]

for feat in cats:

    data[feat] = pd.factorize(data[feat], sort=True)[0]

    

data.head()
# Preparing data for train and test split 

x=data.drop(['id','loss'],1).fillna(value=0)

y=data['loss']
#Train and Test Split 

X_train, X_test, y_train, y_test = train_test_split( x.values, y.values, test_size=0.33, random_state=7)
# fitting model on training data

model = XGBRegressor(max_depth=6, n_estimators=500, learning_rate=0.1, subsample=0.8, colsample_bytree=0.4,

                     min_child_weight = 3,  seed=7)

model.fit(X_train, y_train)

print(model)

#Making predictions

y_pred = model.predict(X_test) 

predictions = [round(value) for value in y_pred]

# evaluat predictions 

actuals = y_test

print(mean_absolute_error(actuals, predictions))

print(model.score(X_test, y_test)) 
Data_Test = read_csv('../input/test.csv')

print(Data_Test.head())

kfeatures = Data_Test.columns

cats = [feat for feat in features if 'cat' in feat]

for feat in cats:

    Data_Test[feat] = pd.factorize(Data_Test[feat], sort=True)[0]

    

Test_X1 = Data_Test.drop(['id'],1).fillna(value=0)

Test_X = Test_X1.values

Data_Test['loss'] = model.predict(Test_X)



Final_Result = Data_Test[['id','loss']]

print(Final_Result.info())

print(Final_Result.head())
#Saving results to csv file

Final_Result.to_csv('result.csv', index=False)