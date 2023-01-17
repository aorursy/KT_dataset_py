import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split
train_data = pd.read_csv('../input/houses_train.csv', index_col=0)

dummy = pd.read_csv('../input/houses_test.csv', index_col=0)

dummy2=pd.concat([train_data,dummy],axis=0) #cheating use test data for the encoder to have unknown cities as well. 
X=X_train = train_data.drop(columns='price')

y=y_train = train_data['price']
train_data.head()
X.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

dummy2['muni_enc']=0

dummy2['muni_enc']=le.fit_transform(dummy2.municipality_name)

X['muni_enc']=0

X['muni_enc']=le.transform(X.municipality_name)

X.drop(['municipality_name'],axis=1,inplace=True)



le2 = preprocessing.LabelEncoder()

X['otn']=0

X['otn']=le2.fit_transform(X.object_type_name)

X.drop(['object_type_name'],axis=1,inplace=True)

import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np



#X, y = X_train,y_train



data_dmatrix = xgb.DMatrix(data=X,label=y)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=X['otn'],random_state=123)





xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.9, learning_rate = 0.05,

                max_depth = 10, alpha = 10, n_estimators = 300)



xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
y_dev_pred = preds#np.random.randint(10000, 2000000, X_test.shape[0])
def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mean_absolute_percentage_error(y_test, y_dev_pred)
X_test = pd.read_csv('./data/houses_test.csv', index_col=0)
from sklearn import preprocessing

#le = preprocessing.LabelEncoder()

X_test['muni_enc']=0

X_test['muni_enc']=le.transform(X_test.municipality_name)

X_test.drop(['municipality_name'],axis=1,inplace=True)

#le2 = preprocessing.LabelEncoder()

X_test['otn']=0

X_test['otn']=le2.transform(X_test.object_type_name)

X_test.drop(['object_type_name'],axis=1,inplace=True)



preds = xg_reg.predict(X_test)
y_test_pred = preds#np.random.randint(10000, 2000000, X_test.shape[0])
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['price'] = y_test_pred
X_test_submission.to_csv('random_submission.csv', header=True, index_label='id')