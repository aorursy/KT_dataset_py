# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading the train and test data from .csv files
data1 = pd.read_csv("../input/minor-project-2020/train.csv");
data2 = pd.read_csv("../input/minor-project-2020/test.csv");
# Count of Each categories that target has in training data
data1[{'id','target'}].groupby(by='target').count()
# The number of unique value each column has
# If a column has only one unique value then it is redundant and we can ommit it
# In our dataset each column has unique data values
data1.nunique()
# Count of the number of NA values in each column
# If a column has too many NA values then we can omit that column or replace NA with apropriate values
# Our training data doesn't have any NA values
data1.isna().sum()
# Check if any row is repeated or duplicated
# The duplicated rows can be dropped fromm the data
# Our Data doesn't have any repetitions
data1.duplicated()
# Create a copy of test data for later use
test_copy = data2.copy()

#drop id column since is not helpful for classification 
data1.drop('id',axis=1,inplace=True)
data2.drop('id',axis=1,inplace=True)

# Create numpy arrays from training and test data
d1 = data1.to_numpy()
d2 = data2.to_numpy()  # d2 will be same as X_test

# Y_train is the last column i.e. target of training data.
# The selection return 2-D array which we convert to a vector
Y_train = d1[:,-1:].flatten()

# X1 is same as X_train
X1 = d1[:,:-1]

# create a common array d appending both X_train and X_test
# Standardisation will be done on this array
d = np.append(X1,d2,axis=0)
# from sklearn.preprocessing import Normalizer

# transformer = Normalizer().fit(d)
# d = transformer.transform(d)

# X_train = d[:X1.shape[0],:]
# x_test = d[X1.shape[0]:,:]

# print(X_train.shape, x_test.shape,len(Y_train))



# Standardisation gave better results than Normalizier and robust scaler
from sklearn.preprocessing import StandardScaler

# Standardise the common array
transformer = StandardScaler().fit(d)
d = transformer.transform(d)

# Retrieve the standardised X_train and X_test from d
X_train = d[:X1.shape[0],:]
x_test = d[X1.shape[0]:,:]

print("X_train shape :",X_train.shape)
print("X_test shape :", x_test.shape)
print("Y_train shape :", len(Y_train))
# Create a correlation matrix for the columns of X_train
# If correlation for any columns is close to 1 or -1 then one of the columns can be dropped
# Our trining data doesn't have any highly correlated columns

mask = np.zeros_like(data1.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set_style('whitegrid')
plt.subplots(figsize = (15,12))
sns.heatmap(data1.corr(),
            mask = mask,
            cmap = 'RdBu', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
from sklearn.linear_model import LogisticRegression

# create Logistic Regression model with parameters obtained as 
# best_params from the following GridSearch
log = LogisticRegression(penalty='l1', solver='liblinear', C=1)

# Train the model and get probabilty values for test data
log.fit(X_train,Y_train)
y_prob = log.predict_proba(x_test)
# This section involves grid search to find the best parameters for logistic regression
# Since this Part takes a lot time for execution, the best parameters returned locally are directly added to
# the model.
# The parametes chosen are based on the sklearn documentation



# from sklearn.model_selection import GridSearchCV

# parameters = {
#     'penalty':['l1','l2','elasticnet','none'],
#     'solver':['lbfgs','newton-cg','liblinear','sag','saga'],
#     'C':[0.5,1,1.5]
# }
# clf = GridSearchCV(log, parameters, verbose=True, n_jobs=4);

# best_clf = clf.fit(X_train,Y_train)

# print(best_clf.best_estimator_)
# y_prob = best.predict_proba(x_test)
# Get the probability of getting 1(Last column) from probability values obtained
prob_1 = y_prob[:,-1:]

# get the ID values from the test data
id1 = test_copy[{'id'}]

# For final submission concat ids with the predicted probability values
ans = pd.concat([id1,pd.DataFrame(prob_1)],axis=1)

# Give names to columns
ans.columns = ['id','target']

# Write to submission.csv file
# The csv contains first column as row number which has to deleted manually to match the submission format
ans.to_csv('submission.csv',index=False)