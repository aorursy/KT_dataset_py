import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.filterwarnings('ignore')
X_train = pd.read_csv("../input/train.csv")

X_test = pd.read_csv("../input/test.csv")
print(X_train.isnull().sum().sum())

print(X_test.isnull().sum().sum())
y_train = X_train['target']

X_train = X_train.drop(['target', 'ID_code'], axis = 1)

submission = X_test['ID_code']

X_test = X_test.drop(['ID_code'], axis = 1)
corr_matrix = X_train.corr(method = 'pearson').abs()



# Select upper triangle of correlation matrix

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.95

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
X_train = X_train.drop(X_train[to_drop], axis = 1)

X_test = X_test.drop(X_test[to_drop], axis = 1)
#from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state = 42) 
# print(X_train.shape)

# print(X_val.shape)

# print(X_test.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(penalty = 'l1', C = 0.1)



# Create regularization penalty space

# penalty = ['l1', 'l2']



# # Create regularization hyperparameter space

# C = np.logspace(0, 4, 10)



# # Create hyperparameter options

# hyperparameters = dict(C = C, penalty = penalty)

# Create grid search using 5-fold cross validation

# from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search

best_model = logistic.fit(X_train, y_train)
# View best hyperparameters

# print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])

# print('Best C:', best_model.best_estimator_.get_params()['C'])
pred = best_model.predict(X_test)
# from sklearn.metrics import accuracy_score

# accuracy_score(y_val, pred)
pred = pd.DataFrame(pred)

pred.columns = ['target']
frames = [submission, pred]
sub = pd.concat(frames, axis = 1)

sub
#export_csv = sub.to_csv (r'C:\Users\ashu1\OneDrive\Desktop\My Projects Data\santander-customer-transaction-prediction\sub.csv', index = None, header=True) 