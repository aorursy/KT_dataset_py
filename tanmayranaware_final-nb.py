from google.colab import drive

drive.mount('/content/drive')


import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from sklearn.preprocessing import scale


df_train=pd.read_csv("drive/My Drive/Train_data.csv")

df_test=pd.read_csv("drive/My Drive/Test_data.csv")
df_train.columns
df_train.describe()
df_test.describe()
df_train.info()
df_test.info()
df_train.isnull().sum()
df_test.isnull().sum()
Y_train=df_train['O/P']

Y_train=Y_train.to_numpy()

print(Y_train.shape)
X_train=df_train.iloc[:,3:18]

X_test=df_test.iloc[:,3:18]
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
rfc=RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',

                      max_depth=40, max_features='auto', max_leaf_nodes=None,

                      max_samples=None, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=5, min_weight_fraction_leaf=0.0,

                      n_estimators=165, n_jobs=None, oob_score=False,

                      random_state=None, verbose=0, warm_start=False)

rfc.fit(X_train,Y_train)
train_predict=rfc.predict(X_train)
test_predict=rfc.predict(X_test)

print(test_predict)
Y_test=pd.read_csv("drive/My Drive/output_true.csv")

y_test=Y_test['PredictedValue']

y_test.to_numpy()

test_index=df_test['Unnamed: 0']
rfc.score(X_test, y_test)
import math

from sklearn.metrics import mean_squared_error
trainScore=math.sqrt(mean_squared_error(Y_train,train_predict))

print(trainScore)
testScore=math.sqrt(mean_squared_error(y_test,test_predict))

print(testScore)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(test_predict)

result.head()
result.to_csv('output22.csv', index=False)