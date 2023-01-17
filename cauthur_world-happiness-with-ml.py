# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
"""
1.Find some relationship between x() and y(Score) variable - Heatmap
2. Do some ML to find some accurate model - Linear Regression, AdaBoost, R2 score, MSE, GridSearch
3. Find feature that affect most on y(Score) variable - RandomForest Regressor feature importance
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("../input/world-happiness/2019.csv")
display(df)
correlation = df.corr()
plt.figure(figsize=(16,16))
sns.heatmap(correlation,fmt=".2f",annot=True,cmap="YlGnBu")
null_value = df.isnull().sum()
print(null_value)
val = df.columns[2:].to_list()

plt.figure(figsize=(60,15))
for i in range(1,8):
    plt.subplot(1,7,i)
    plt.hist(df[val[i-1]])
    plt.title("{}".format(val[i-1]),fontsize=25)
df_log = df
df_log["Social support"] = np.log(df_log["Social support"] + 1)
df_log["Perceptions of corruption"] = np.log(df_log["Perceptions of corruption"] + 1)
display(df_log)
X = df_log.iloc[:,3:]
y = df_log.Score
X_train,X_test,y_train,y_test = train_test_split(X,y)
li_model = LinearRegression()
Ada_model = AdaBoostRegressor()


li_model.fit(X_train,y_train)
Ada_model.fit(X_train,y_train)


li_preds = li_model.predict(X_test)
Ada_preds = Ada_model.predict(X_test)

li_mse = mean_squared_error(y_test,li_preds)
Ada_mse = mean_squared_error(y_test,Ada_preds)
li_r2 = r2_score(y_test,li_preds)
Ada_r2 = r2_score(y_test,Ada_preds)

Error = pd.DataFrame({
    "linear_model" : [li_mse,li_r2],
    "Ada_model" : [Ada_mse,Ada_r2]
   
},index = ["MSE","R2"])
display(Error)

params = {
    "n_estimators" : [50,60,70,80,90,100,120,140,160,180,200],
    "learning_rate" : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
}
scorer = make_scorer(r2_score)
gs = GridSearchCV(estimator=AdaBoostRegressor(),param_grid=params,scoring=scorer,n_jobs=4)
gs_fit = gs.fit(X_train,y_train)
gs_best = gs_fit.best_estimator_
gs_best.fit(X_train,y_train)
gs_best_preds = gs_best.predict(X_test)
gs_r2 = r2_score(y_test,gs_best_preds)

print("Best estimator parms : {}".format(gs_best.get_params))
print("Best estimator r2 score : {}".format(gs_r2))
RFR = RandomForestRegressor()
RFR.fit(X_train,y_train)
feature_importance = np.sort(np.round(RFR.feature_importances_*100,1))
df_feature = pd.DataFrame({
    "importance" : feature_importance
},index = X_train.columns.to_list())

display(df_feature)

plt.figure(figsize=(15,15))
plt.barh(df_feature.index.to_list(),df_feature.importance)
plt.title("Feature importance",fontsize=15)