import pandas as pd
from sklearn.datasets import load_boston
boston_data = load_boston()
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston['MEDV'] = boston_data.target
boston.head()

X = boston.drop(columns=['RAD','MEDV'])
Y = boston['MEDV']
len(Y)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size = 0.1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(y_test.shape)
y_test.reset_index(drop=True, inplace=True)
y_test
Train = pd.concat([X_train,Y_train], axis=1)
Train.reset_index(drop=True, inplace=True)
Test = X_test
Test.reset_index(drop=True, inplace=True)
Train

X = Train.drop(columns='MEDV')
Y = Train['MEDV']
Y
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

y_test_predict = lin_model.predict(X_test)


rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print('RMSE is {}'.format(rmse))
final = lin_model.predict(Test)
final

rmse = (np.sqrt(mean_squared_error(y_test, final)))
print('RMSE is {}'.format(rmse))
Test
X
# X = Train.drop(columns='MEDV')
# Y = Train['MEDV']
from sklearn.model_selection import KFold 
score = []
final = []
kfold = KFold(n_splits=5, random_state=24, shuffle=True)

for train, test in kfold.split(X):
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    y_test_predict = lin_model.predict(X_test)
    
    final_pred = lin_model.predict(Test)
    final.append(final_pred)
    
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    print(rmse)
    score.append(rmse)
        
        
average_score = np.mean(score)
print('The average RMSE is ', average_score)
final_result = np.mean(final,0)
final_result
for i in range(0,len(final)):
    
    rmse = (np.sqrt(mean_squared_error(y_test, final[i])))
    print('RMSE is {}'.format(rmse))
from xgboost import XGBRegressor
lin_model = XGBRegressor()
lin_model.fit(X_train, Y_train)
y_test_predict = lin_model.predict(X_test)


rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print('RMSE is {}'.format(rmse))

test_model = XGBRegressor(
            eta = 0.03,
            n_estimators = 1500 
)
#model.fit(X_train, y_train)
test_model.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)
a = test_model.best_iteration

from xgboost import XGBRegressor
lin_model = XGBRegressor( eta = 0.03,
            n_estimators = a )
lin_model.fit(X_train, Y_train)
y_test_predict = lin_model.predict(X_test)


rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
print('RMSE is {}'.format(rmse))

# RMSE is 3.2106485585832782
final = lin_model.predict(Test)


rmse = (np.sqrt(mean_squared_error(y_test, final)))
print('RMSE is {}'.format(rmse))

from sklearn.model_selection import KFold 
score = []
final = []
kfold = KFold(n_splits=5, random_state=24, shuffle=True)

for train, test in kfold.split(X):
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    
    test_model = XGBRegressor(
            eta = 0.03,
            n_estimators = 1500 
    )
    #model.fit(X_train, y_train)
    test_model.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)
    a = test_model.best_iteration
    
    lin_model = XGBRegressor( eta = 0.03,
            n_estimators = a )
    lin_model.fit(X_train, Y_train)
    y_test_predict = lin_model.predict(X_test)
    
    
    final_pred = lin_model.predict(Test)
    final.append(final_pred)
    
    rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
    print(rmse)
    score.append(rmse)
        
        
average_score = np.mean(score)
print('The average RMSE is ', average_score)
final
for i in range(0,len(final)):
    
    rmse = (np.sqrt(mean_squared_error(y_test, final[i])))
    print('RMSE is {}'.format(rmse))
    
#RMSE is 2.467268931228599
finals = np.mean(final,0)
rmse = (np.sqrt(mean_squared_error(y_test, finals)))
print('RMSE is {}'.format(rmse))
    
from sklearn.datasets import load_wine
wine_dataset = load_wine()
wine = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)

wine['quality'] = wine_dataset.target
wine.head()
wine = wine[wine.quality !=2]
X = wine.drop(columns='quality')
Y = wine['quality']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=32)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)
from sklearn.metrics import   f1_score, accuracy_score
accuracy_score(pred, Y_test)

from sklearn.model_selection import StratifiedKFold
# kfold = StratifiedKFold(n_splits=10, random_state=27, shuffle=True)
kfold = KFold(n_splits=7, random_state=24, shuffle=True)
score = []
for train, test in kfold.split(X):
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    
    model = LogisticRegression(max_iter = 10000)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    f1 = accuracy_score(pred, Y_test)
    print(f1)
    score.append(f1)
        
average_score = np.mean(score)
print('The average accuracy is ', average_score)    
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, random_state=27, shuffle=True)
# kfold = KFold(n_splits=7, random_state=24, shuffle=True)
score = []
for train, test in kfold.split(X,Y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    Y_train, Y_test = Y[train], Y[test]
    
    model = LogisticRegression(max_iter = 10000)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    f1 = accuracy_score(pred, Y_test)
    print(f1)
    score.append(f1)
        
average_score = np.mean(score)
print('The average accuracy is ', average_score)    