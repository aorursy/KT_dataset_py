import numpy as np

import pandas as pd
re = pd.read_csv('../input/RE.csv')
re.head()
re.shape
re.columns
re1 = re.drop(['No', 'X1 transaction date'],1)
re1.head()
re1.isnull().sum()
re1.info()
re1.describe()
re1['X4 number of convenience stores'].unique()
re1.skew()
re1.kurt()
print(np.median(re1['X2 house age']))

print(np.median(re1['X3 distance to the nearest MRT station']))

print(np.median(re1['X4 number of convenience stores']))

print(np.median(re1['X5 latitude']))

print(np.median(re1['X6 longitude']))

print(np.median(re1['Y house price of unit area']))
np.mean(re1)
import seaborn as sns

sns.pairplot(re1, diag_kind = 'kde')
X = re1.drop('Y house price of unit area',1)

y = re1['Y house price of unit area']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100) 
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state = 100)
dt = DecisionTreeRegressor(max_depth = 4, random_state = 100)

br = BaggingRegressor(base_estimator = dt, n_estimators = 100, random_state = 100)
rf = RandomForestRegressor()

br_fit = br.fit(X_train,y_train)
br_fit
br1 = BaggingRegressor(base_estimator = rf, random_state = 100)
br1_fit = br1.fit(X_train, y_train)
y_pred = br_fit.predict(X_test) #decision tree
y_pred = br1_fit.predict(X_test)#random forest
# DECISION TREE BASE ESTIMATOR
print(br.score(X_train,y_train)) 
print(br.score(X_test,y_test))
# RANDOM FOREST BASE ESTIMATOR
print(br1.score(X_train,y_train))

print(br1.score(X_test,y_test))
# USING KFOLD

from sklearn.model_selection import cross_val_score

res_br=cross_val_score(br,X_train,y_train.ravel(),cv=kfold)

print(np.mean(res_br))
from sklearn.model_selection import cross_val_score

res_br1=cross_val_score(br1,X_train,y_train.ravel(),cv=kfold)

print(np.mean(res_br1))