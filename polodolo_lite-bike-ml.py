import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv",parse_dates=["datetime"])
train.head()
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv",parse_dates=["datetime"])
test.head()
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
train.shape
test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
test.shape
categorical_feature = ["season","holiday","workingday","weather","dayofweek","month","year","hour"]
for var in categorical_feature:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")
train.info()
feature = ["season","holiday","workingday","weather","dayofweek","year","hour","temp","atemp","humidity"]
X_train = train[feature]
X_test = test[feature]
X_train.head()
Y_train = train["count"]
Y_train.head()
model = RandomForestRegressor(n_estimators=500)

Y_train_log = np.log1p(Y_train)
model.fit(X_train,Y_train_log)

result = model.predict(X_test)
sub = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv")
sub.head()
sub["count"] = np.exp(result)
sub.head()
sub.to_csv("lite_test.csv",index=False)