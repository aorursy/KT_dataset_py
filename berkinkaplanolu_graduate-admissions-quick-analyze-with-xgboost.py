import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
data.head()
data.info()
from sklearn.preprocessing import MinMaxScaler # Scaling values.
mms = MinMaxScaler()
scaledData = pd.DataFrame(mms.fit_transform(data.iloc[:,1:7]), columns = ["GRE Score","TOEFL Score","University Rating","SOP","LOR","CGPA"])
newData = pd.concat((scaledData,data.iloc[:,7:9]),axis = 1)
newData
X = newData.iloc[:,:-1]
y = newData.iloc[:,-1]
from sklearn.model_selection import train_test_split ,GridSearchCV 
X_train, X_test, y_train ,y_test = train_test_split(X,y,random_state = 44, test_size = 0.2) #Splitting data.
from xgboost import XGBRegressor
XGBR = XGBRegressor()
modelParams = {"max_depth": [2,3,5,10,30],
              "subsample": [0.5,0.75,1],
              "colsample_bytree":[0.5,0.75,1],
              "colsample_bylevel":[0.5,0.75,1],
              "min_child_weight": [1,5,25],
              "n_estimators": [10,50,100,250,500],
              "learning_rate":[0.01,0.1,0.25]} 
XGBGridSearch = GridSearchCV(XGBR, modelParams,verbose = 2,n_jobs = -1,cv = 5) #n_jobs = -1 means use all cores for training
XGBGridSearch.fit(X_train,y_train)
XGBGridSearch.best_params_
XGBR2 = XGBRegressor(colsample_bylevel= 0.5,
 colsample_bytree= 0.75,
 learning_rate= 0.01,
 max_depth= 2,
 min_child_weight= 5,
 n_estimators= 500,
 subsample= 0.75) # Training with best parameters.
XGBR2.fit(X_train,y_train)
y_pred = XGBR2.predict(X_test)
XGBR2.fit(X_train,y_train)
y_pred2 = XGBR2.predict(X_train)
from sklearn.metrics import r2_score
print(f"test set r2 value:{r2_score(y_test, y_pred)} train set r2 value{r2_score(y_train, y_pred2)}")
print(pd.DataFrame(np.vstack((y_test.to_numpy(),y_pred)).T,columns = ["Actual values","Predicted values"]))
