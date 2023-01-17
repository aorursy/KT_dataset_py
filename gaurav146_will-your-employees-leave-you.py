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
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
train = pd.read_csv("/kaggle/input/hacker-earth-will-your-employees-leave-you/Train.csv")
test = pd.read_csv("/kaggle/input/hacker-earth-will-your-employees-leave-you/Test.csv")
submit = pd.read_csv("/kaggle/input/hacker-earth-will-your-employees-leave-you/sample_submission.csv")
train.shape, test.shape, submit.shape
test_IDs = test.Employee_ID
train.head(4)
i = train.pop("Employee_ID")
i = test.pop("Employee_ID")
# i = test.pop("Hometown")
# i = train.pop("Hometown")
train["Mark"] = 1
test["Mark"] = 0
data = pd.concat([train, test])
data.shape
data.tail()
data = data.reset_index()
data["Age"].fillna(np.ceil(data.Age.mean()), inplace=True)
data["Time_of_service"].fillna(np.ceil(data.Time_of_service.mean()), inplace=True)
data["Work_Life_balance"].fillna(np.ceil(data.Work_Life_balance.mean()), inplace=True)
data.isnull().sum()
data.VAR2.fillna(data.VAR2.mean(), inplace=True)
data.VAR4.fillna(data.VAR4.mean(), inplace=True)
data.Pay_Scale.fillna(np.ceil(data.Work_Life_balance.mean()), inplace=True)
data.shape
data.head(3)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data.Gender  = le.fit_transform(data.Gender)
data.Relationship_Status  = le.fit_transform(data.Relationship_Status)
data.Hometown  = le.fit_transform(data.Hometown)
data.Unit  = le.fit_transform(data.Unit)
data.Decision_skill_possess = le.fit_transform(data.Decision_skill_possess)
data.Compensation_and_Benefits = le.fit_transform(data.Compensation_and_Benefits)
# dum_df = pd.get_dummies(data.Gender)
# data = data.join(dum_df)
# dum_df = pd.get_dummies(data.Relationship_Status)
# data = data.join(dum_df)
# dum_df = pd.get_dummies(data.Hometown)
# data = data.join(dum_df)
# dum_df = pd.get_dummies(data.Unit)
# data = data.join(dum_df)
# dum_df = pd.get_dummies(data.Decision_skill_possess)
# data = data.join(dum_df)
# dum_df = pd.get_dummies(data.Compensation_and_Benefits)
# data = data.join(dum_df)
data.head(2)
data.shape
data = data.drop(['Gender','Relationship_Status','Hometown','Unit','Decision_skill_possess','Compensation_and_Benefits'],axis=1)
# data = data.drop(['Gender','Unit','Compensation_and_Benefits'],axis=1)
data = data.drop(['F','Single','Washington','Sales','Directive','type4'],axis=1)
# data = data.drop(['F','Sales','type4'],axis=1)
data.dtypes
data = data.reset_index()
data = data.drop(['level_0','index'],axis=1)
data.head()
train = data[data.Mark == 1]
test = data[data.Mark == 0]
train.shape, test.shape
test.columns
test = test.reset_index()
test = test.drop(['index'],axis=1)
train = train.drop(['Mark'],axis=1)
test = test.drop(['Mark','Attrition_rate'],axis=1)
X = train.drop('Attrition_rate',axis=1)
Y = train.Attrition_rate
X.shape, test.shape
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
x_train.shape, y_train.shape, x_test.shape
X.shape, Y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train.shape, test.shape
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
my_test = sc.transform(test)
from sklearn.decomposition import PCA

pca = PCA()
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
explained_variance
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)

X_train1.shape,X_test1.shape
# 22 24 27 30 60
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
# rgr = GradientBoostingRegressor()
# from sklearn.linear_model import Ridge
# rgr = Ridge(alpha=1.0)

# rgr = XGBRegressor(max_depth=0,subsample=0.1, random_state=0)
# rgr = LinearRegression()
# from sklearn.linear_model import ElasticNet
# rgr = ElasticNet(random_state=0)

from sklearn.ensemble import AdaBoostRegressor
rgr = AdaBoostRegressor(random_state=0, n_estimators=1)
rgr.fit(X_train1, y_train)

# Predicting the Test set results
y_pred = rgr.predict(X_test1)
metrics.mean_squared_error(y_pred, y_test)
test_modi = sc.transform(test)
test1 = pca.transform(test_modi)
res = rgr.predict(test1)
subm = pd.DataFrame()
subm["Employee_ID"]=test_IDs
subm["Attrition_rate"] = res
subm.to_csv("result41.csv",index=False)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
num_estimators = [10,100]
learn_rates = [0.01, 0.7]
max_depths = [1, 6]
min_samples_leaf = [5,10]
min_samples_split = [5,10]

param_grid = {'n_estimators': num_estimators,
              'learning_rate': learn_rates,
              'max_depth': max_depths,
              'min_samples_leaf': min_samples_leaf,
              'min_samples_split': min_samples_split}

random_search =RandomizedSearchCV(GradientBoostingRegressor(), param_grid, random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)

random_search.fit(X_train1, y_train)
random_search.best_params_
gboost_score=random_search.score(X_train1, y_train)
gboost_score
print("Sleeping...")
time.sleep(10*60)
print("Done..!")
