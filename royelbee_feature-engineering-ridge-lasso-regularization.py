# Import all necessary librarys 

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler 

# In here we use a large dataset with 133 features and 114321 records
data = pd.read_csv('/kaggle/input/paribas_train.csv')
data.shape
data.head()
# Lets take down only numerical variable 
# After removing all categorical features total numerical features is 114

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_val = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_val]
data.shape
# Now seperate Train and Test sets 

X_train, X_test, y_train, y_test = train_test_split(
                                data.drop(labels=['target', 'ID'], axis=1),
                                data['target'],
                                test_size=.3, random_state=0)
X_train.shape, X_test.shape
# Create linear model 
scaller = StandardScaler()
scaller.fit(X_train.fillna(0))
# select lasso l1 properly 

sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel.fit(scaller.transform(X_train.fillna(0)), y_train)
data.head()
# Lets take down only numerical variable 
# After removing all categorical features total numerical features is 114

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_val = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_val]
data.shape
# Now seperate Train and Test sets 

X_train, X_test, y_train, y_test = train_test_split(
                                data.drop(labels=['target', 'ID'], axis=1),
                                data['target'],
                                test_size=.3, random_state=0)
X_train.shape, X_test.shape
# select lasso l1 properly 

sel = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
sel.fit(scaller.transform(X_train.fillna(0)), y_train)
# Okay lets visualize those features what we keep 
sel.get_support()
# Now, make a list with selected featires 

selected_features = X_train.columns[(sel.get_support())]

print('Total Features    = ', format(X_train.shape[1]))
print('Selected Features = ', format(len(selected_features)))

coeffieient = format(np.sum(sel.estimator_.coef_ == 0))
print('Features with coefficients shrank to zero =',coeffieient)

# Now find the thoese features (22) which we are going to remove

removed_features = X_train.columns[(sel.estimator_.coef_ == 0).ravel().tolist()]
removed_features
# Now we have to remove the features from training and testing set 
# After removing 22 features we have 90 features remain both train and test set

X_train_selected = sel.transform(X_train.fillna(0))
X_test_selected = sel.transform(X_test.fillna(0))
  
X_train_selected.shape, X_test_selected.shape
# Now seperate Train and Test sets 

X_train, X_test, y_train, y_test = train_test_split(
                                data.drop(labels=['target', 'ID'], axis=1),
                                data['target'],
                                test_size=.3, random_state=0)
X_train.shape, X_test.shape
# Create linear model 
scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

ridge = SelectFromModel(LogisticRegression(C=1, penalty='l2'))
ridge.fit(scaler.transform(X_train.fillna(0)), y_train)
# Now, make a list with selected featires 

selected_features = X_train.columns[(ridge.get_support())]

print('Total Features    = ', format(X_train.shape[1]))
print('Selected Features = ', format(len(selected_features)))

coeffieient = format(np.sum(ridge.estimator_.coef_ == 0))
print('Features with coefficients shrank to zero =',coeffieient)

selected_features
