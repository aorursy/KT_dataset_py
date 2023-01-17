import numpy as np
import pandas as pd
import os

data = pd.pandas.read_csv("../input/train_data.csv")

data.head()

data = data.drop( ['Photo_filename'], axis=1)
data.head()
target = data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18']]

data = data.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18'], axis=1)
data.head()

data.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data_scaled = data

data_scaled[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']] = scaler.fit_transform(data_scaled[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']])

d = {1 : 0, 2 : 1}

data_scaled['C'] = data_scaled['C'].map(d)

data_scaled.head()
numeric_cols = [ 'C', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7' ]
x_num = data_scaled[ numeric_cols ].as_matrix()
print (x_num)
cat = data_scaled.drop( numeric_cols, axis = 1 )

cat.head()
cat.fillna( 'NA', inplace = True )
x_cat = cat.to_dict( orient = 'records' )

from sklearn.feature_extraction import DictVectorizer as DV

vectorizer = DV( sparse = False )
vec_x_cat = vectorizer.fit_transform( x_cat )
x_train_values = np.hstack(( x_num, vec_x_cat ))

print (x_num.shape)
print (vec_x_cat.shape)
print (x_train_values.shape)
from sklearn.model_selection import train_test_split, RandomizedSearchCV

X_train, X_valid, Y_train, Y_valid = train_test_split(x_train_values, target.values,
                                                     test_size=.2, random_state=17)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import scipy.stats as st
from sklearn.model_selection import GridSearchCV

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)

params={
    'estimator__n_estimators': [1000,2000,3000],
    'estimator__reg_alpha': [0.01, 0.02, 0.03, 0.04]
}


xgb = XGBRegressor(nthreads=-1, max_depth=3, subsample=0.7, colsample_bytree=0.5)  

xgb = MultiOutputRegressor(xgb)
grid = GridSearchCV(xgb, params, n_jobs=5) 
grid.fit(X_train, Y_train)

print (grid.best_estimator_)

from sklearn.metrics import mean_absolute_error

Y_pred = grid.best_estimator_.predict(X_valid)

print (mean_absolute_error(Y_valid, Y_pred))