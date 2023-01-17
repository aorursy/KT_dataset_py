import numpy as np

import pandas as pd



dfTrain = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

dfTest = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")
dfTrain['Date'] = dfTrain['Date'].apply(lambda x: int(x.replace('-','')))

dfTest['Date'] = dfTest['Date'].apply(lambda x: int(x.replace('-','')))
dfTrain = dfTrain.drop(columns=['Province/State'])
dfTrain.isnull().sum()
dfTrain.head()
dfTrain.dtypes
#Asign columns for training and testing

X_train =dfTrain[['Lat', 'Long', 'Date']]

y1_train = dfTrain[['ConfirmedCases']]

y2_train = dfTrain[['Fatalities']]

X_test = dfTest[['Lat', 'Long', 'Date']]
#We are going to use Random Forest classifier for the forecast

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200)
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                      max_depth=None, max_features='auto', max_leaf_nodes=None, 

                      n_estimators=150, random_state=None, n_jobs=1, verbose=0)
model.fit(X_train,y1_train)

pred1_rf = model.predict(X_test)

pred1_rf = pd.DataFrame(pred1_rf)

pred1_rf.columns = ["ConfirmedCases_prediction"]
pred1_rf.head()
model.fit(X_train,y2_train)

pred2_rf = model.predict(X_test)

pred2_rf = pd.DataFrame(pred2_rf)

pred2_rf.columns = ["Fatalities_prediction"]
pred2_rf.head()
from xgboost import XGBRegressor

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import r2_score



from sklearn.metrics import mean_absolute_error

model2 = XGBRegressor(n_estimators=1000,learning_rate=0.1,objective='reg:squarederror')
y1_train = y1_train.replace(np.nan, 0)

model2.fit(X_train,y1_train)

pred1_xgb = model2.predict(X_test)

pred1_xgb = pd.DataFrame(pred1_xgb)

pred1_xgb.columns = ["ConfirmedCases_prediction"]
pred1_xgb.head()
y2_train = y2_train.replace(np.nan, 0)

model2.fit(X_train,y2_train)

pred2_xgb = model2.predict(X_test)

pred2_xgb = pd.DataFrame(pred2_xgb)

pred2_xgb.columns = ["Fatalities_prediction"]
pred2_xgb.head()
submissionOriginal = pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
submissionOriginal
submissionOriginal["Fatalities"] = pred2_xgb["Fatalities_prediction"]

pred2_xgb.drop("Fatalities_prediction", axis=1, inplace=True)