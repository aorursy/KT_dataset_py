import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/Admission_Predict.csv')

data.info()

data.head(5)
corr_matrix = data.corr()

plt.figure(figsize=(20,10))

sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",annot_kws={'size':16})

plt.show()
corr_matrix["Chance of Admit "].sort_values(ascending=False)
X,Y = data.iloc[:,1:-1].values,data.iloc[:,-1].values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.2, random_state = 33)
from sklearn.tree import DecisionTreeRegressor

dt_reg = DecisionTreeRegressor(random_state=33)

dt_reg.fit(x_train,y_train)
y_train_predic = dt_reg.predict(x_train)

y_test_dt = dt_reg.predict(x_test)

from sklearn.metrics import mean_squared_error,r2_score

print("Decision Tree train_error is:",mean_squared_error(y_train,y_train_predic))

print("Decision Tree test_error is:",mean_squared_error(y_test,y_test_dt))
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators=500,random_state=33,bootstrap=True,n_jobs=-1)

rf_reg.fit(x_train,y_train)

y_test_rf = rf_reg.predict(x_test)

print("RandomForest train_error is:",mean_squared_error(y_train,rf_reg.predict(x_train)))

print("RandomForest test_error is:",mean_squared_error(y_test,y_test_rf))
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(n_estimators=500,learning_rate=0.5,random_state=33)

ada_reg.fit(x_train,y_train)

y_test_ada = ada_reg.predict(x_test)

print("AdaBoost test_error is:",mean_squared_error(y_test,y_test_ada))
from sklearn.ensemble import GradientBoostingRegressor

grbt_reg = GradientBoostingRegressor(max_depth=2,n_estimators=500,random_state=33)

grbt_reg.fit(x_train,y_train)



errors = np.zeros((500,1))

i = 0

for y_pred in grbt_reg.staged_predict(x_test):

    errors[i] = mean_squared_error(y_test,y_pred)

    i = i + 1

    #print(y_pred)



best_n_estimator = np.argmin(errors)



plt.plot(errors)

plt.xlabel('number of trees');plt.ylabel('RMSE');plt.show()



grbt_reg_best = GradientBoostingRegressor(max_depth=2,n_estimators=best_n_estimator)

grbt_reg_best.fit(x_train,y_train)

y_test_gbrt = grbt_reg_best.predict(x_test)



print("GBR test_error is:",mean_squared_error(y_test,y_test_gbrt))
plt.hist(y_test-y_test_gbrt,bins=20);

plt.xlabel('Prediction Error');plt.ylabel('Frequency');plt.show()
import xgboost as xgb

xgb_reg = xgb.XGBRegressor(random_state=33,num_parallel_tree=500,learning_rate=0.05,early_stopping_rounds=10,max_depth=2)

xgb_reg.fit(x_train,y_train)

y_test_xgb = xgb_reg.predict(x_test)



print("GBR test_error is:",mean_squared_error(y_test,y_test_xgb))
plt.hist(y_test-y_test_xgb,bins=20);

plt.xlabel('Prediction Error');plt.ylabel('Frequency');plt.show()