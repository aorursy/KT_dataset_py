%matplotlib inline
# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
# get training & test csv files as a DataFrame
train_df = pd.read_csv("../input/train.csv" )
test_df    = pd.read_csv("../input/test.csv")
# There are some columns with non-numerical values(i.e. dtype='object'),
# So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.

from sklearn import preprocessing

for f in train_df.columns:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_df[f].values) + list(test_df[f].values)))
        train_df[f] = lbl.transform(list(train_df[f].values))
        test_df[f]       = lbl.transform(list(test_df[f].values))
# fill NaN values

for f in train_df.columns:
    if f == "Response": continue
    if train_df[f].dtype == 'float64':
        train_df[f].fillna(train_df[f].mean(), inplace=True)
        test_df[f].fillna(test_df[f].mean(), inplace=True)
    else:
        train_df[f].fillna(train_df[f].median(), inplace=True)
        test_df[f].fillna(test_df[f].median(), inplace=True)
# define training and testing sets
# Remove Height and Weight. Rely on BMI

X_train = train_df.drop(["Response", "Id", "Ht", "Wt"],axis=1)
y_train = train_df["Response"]
X_test  = test_df.drop(["Id", "Ht", "Wt"],axis=1).copy()
# modify response values so that range of values is from 0-7 instead of 1-8
y_train = y_train - 1
# Find the features that really matter in data set using Random Forest Classifier

feat_labels = X_train.columns
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
importances
# identify the list of top features

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
# Use only top features
X_train = forest.transform(X_train, threshold=.008)
X_test = forest.transform(X_test, threshold=.008)

# Define different models
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

lr = LogisticRegression(C=1.0, random_state=0)

knn = KNeighborsClassifier(n_neighbors=15, p=2, metric='minkowski')

gnb = GaussianNB()

lsvc = LinearSVC()

lsvm = SVC(kernel='linear')
# Perform pre-processing to determine optimal data set size and tune model parameters


# Determine optimal training data set size using learning curve methods
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

    
train_sizes, train_scores, test_scores = learning_curve(estimator=knn, X=X_train, y=y_train, 
                                                        train_sizes=np.linspace(0.1, 1.0, 5), cv=5)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.1, 1.2])
plt.show()
from sklearn.learning_curve import validation_curve


param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(estimator=lr, X=X_train, y=y_train, param_name='C',
                                            param_range=param_range, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.4, 0.6])
plt.show()
nn_range = [10, 15, 20, 25, 30, 35]
train_scores, test_scores = validation_curve(estimator=knn, X=X_train, y=y_train, param_name='n_neighbors',
                                            param_range=nn_range, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(nn_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(nn_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(nn_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(nn_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.20, 0.65])
plt.show()
ne_range = [10, 100, 1000]
train_scores, test_scores = validation_curve(estimator=forest, X=X_train, y=y_train, param_name='n_estimators',
                                            param_range=ne_range, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(ne_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(ne_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(ne_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(ne_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.40, 0.85])
plt.show()
# in addition to the original data sets for training (train_orig)and testing (test_orig)
# split train_orig data into training and testing sets randomly so we can obtain a practice test set with outcomes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.30, random_state=0)
# Xgboost 

params = {"objective": "multi:softmax", "num_class": 8}

T_train_xgb = xgb.DMatrix(X_train, y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
y_pred = gbm.predict(X_test_xgb)
# change values back to range of values is from 1-8 instead of 0-7

y_pred = y_pred + 1
y_pred = y_pred.astype(int)
# Create submission

output = pd.DataFrame({
        "Id": test_df["Id"],
        "Response": y_pred
    })
output.to_csv("../input/output.csv", index=False)