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
testset=pd.read_csv('../input/summeranalytics2020/test.csv')
trainset=pd.read_csv('../input/summeranalytics2020/train.csv',index_col=False)
# trainset.set_index("Id", inplace=True)
ss=pd.read_csv('../input/summeranalytics2020/Sample_submission.csv')
# testset.set_index("Id",inplace=True)
trainset.head()



testset = testset.drop(['EmployeeNumber','Behaviour'], axis = 1)
trainset = trainset.drop(['EmployeeNumber','Behaviour'], axis = 1)
trainset.head()
[trainset.shape, testset.shape]
trainset.dtypes
from sklearn.preprocessing import LabelEncoder

for column in testset.columns:
        if testset[column].dtype == np.number:
            continue
        testset[column] = LabelEncoder().fit_transform(testset[column])
for column in trainset.columns:
        if trainset[column].dtype == np.number:
            continue
        trainset[column] = LabelEncoder().fit_transform(trainset[column])
trainset.describe()
trainset.head()
# trainset.isna().any() # NO NA found in the data
trainset["Attrition"].value_counts()
trainset.corr()
import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize = (16,10))
heatmap=sb.heatmap(trainset.corr(),cmap="YlGnBu",annot=True)
trainset_corr=trainset.corr()
# Highly correlated features with pearson correlation between 0.7-0.8 or greater
for column in trainset_corr.columns:
    trainset_corr[trainset_corr[column]>=0.7]
trainset_corr.loc[trainset_corr['YearsAtCompany']>0.7]
testset.columns
from sklearn.preprocessing import StandardScaler

features = ['Age', 'DistanceFromHome','MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears',
       'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
       'YearsSinceLastPromotion', 'YearsWithCurrManager']
# Separating out the features
x = trainset.loc[:, features].values
xt = testset.loc[:, features].values


# Standardizing the features
x = StandardScaler().fit_transform(x)
df = pd.DataFrame(x, columns=features).astype('float64')

xt = StandardScaler().fit_transform(xt)
dft = pd.DataFrame(xt, columns=features).astype('float64')

# Remove categorical columns 
num_X_train = trainset.drop(features, axis=1)
num_X_test = testset.drop(features, axis=1)

# Add categorical columns to numerical features
trainset = pd.concat([num_X_train, df], axis=1)
testset = pd.concat([num_X_test, dft], axis=1)
# # OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)
# df.reset_index().set_index('index', drop=False)
trainset.set_index("Id", inplace=True)
testset.set_index("Id", inplace=True)

# df
# num_X_train
testset
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold

# HERE WE SPLIT DATA INTO TRAIN TEST SPLIT
X_train,X_test,y_train,y_test = train_test_split(trainset.drop('Attrition',axis = 1),trainset['Attrition'],test_size = 0.2,random_state = 2)
Lr=LogisticRegression(solver='lbfgs',max_iter=5000,C=0.5,penalty='l2',random_state=1)
Lr.fit(X_train,y_train)
[X_train.shape, X_test.shape, y_train.shape, y_test.shape]
roc_auc_logreg_train = cross_val_score(Lr, X_train, y_train, cv = 11, scoring = 'roc_auc').mean()
roc_auc_logreg_test = cross_val_predict(Lr, X_test,y_test, cv = 11, method='predict')
roc_auc_logreg_train
# roc_auc_logreg_test
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier


import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# Train multiple models with various hyperparameters using the training set, select the model and hyperparameters that perform best on the validation set.
# Once model type and hyperparameters have been selected, train final model using these hyperparameters on the full training set, the generalized error is finally measured on the test set.
X_train,X_test,y_train,y_test = train_test_split(trainset.drop('Attrition',axis = 1),trainset['Attrition'],test_size = 0.2,random_state = 1)

# StratifiedKFold class performs stratified sampling to produce folds that contain a representative ratio of each class.
cv = StratifiedKFold(n_splits=10, shuffle = False, random_state = 76)

# Logistic Regression
clf_logreg = LogisticRegression()
# fit model
clf_logreg.fit(X_train, y_train)
# Make class predictions for the validation set.
y_pred_class_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv)
# predicted probabilities for class 1, probabilities of positive class
y_pred_prob_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_logreg_class1 = y_pred_prob_logreg[:, 1]

# SGD Classifier
clf_SGD = SGDClassifier()
# fit model
clf_SGD.fit(X_train, y_train)
# make class predictions for the validation set
y_pred_class_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv)
# predicted probabilities for class 1
y_pred_prob_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv, method="decision_function")

# Random Forest Classifier
clf_rfc = RandomForestClassifier(n_estimators=100)
# fit model
clf_rfc.fit(X_train, y_train)
# make class predictions for the validation set
y_pred_class_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv)
# predicted probabilities for class 1
y_pred_prob_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_rfc_class1 = y_pred_prob_rfc[:, 1]
y_pred_prob_logreg_class1 = y_pred_prob_logreg[:, 1]
y_pred_prob_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv, method="decision_function")
y_pred_prob_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv, method="predict_proba")
y_pred_prob_logreg_class1, y_pred_prob_SGD, y_pred_prob_rfc
from sklearn.base import BaseEstimator
import numpy as np

class BaseClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
base_clf = BaseClassifier()
cross_val_score(base_clf, X_train, y_train, cv=10, scoring="accuracy").mean()


# Method 2
# calculate null accuracy (for binary / multi-class classification problems)
# null_accuracy = y_train.value_counts().head(1) / len(y_train)
# calculate accuracy

acc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'accuracy').mean()
acc_SGD = cross_val_score(clf_SGD, X_train, y_train, cv = cv, scoring = 'accuracy').mean()
acc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'accuracy').mean()

acc_logreg, acc_SGD, acc_rfc
# calculate logloss

logloss_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()
logloss_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()

# SGDClassifier's hinge loss doesn't support probability estimates.
# We can set SGDClassifier as the base estimator in Scikit-learn's CalibratedClassifierCV, which will generate probability estimates.

from sklearn.calibration import CalibratedClassifierCV

new_clf_SGD = CalibratedClassifierCV(clf_SGD)
new_clf_SGD.fit(X_train, y_train)
logloss_SGD = cross_val_score(new_clf_SGD, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()

logloss_logreg, logloss_SGD, logloss_rfc
roc_auc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()
roc_auc_SGD = cross_val_score(clf_SGD, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()
roc_auc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()

roc_auc_logreg, roc_auc_SGD, roc_auc_rfc
# IMPORTANT: first argument is true values, second argument is predicted probabilities

# we pass y_test and y_pred_prob
# we do not use y_pred_class, because it will give incorrect results without generating an error
# roc_curve returns 3 objects false positive rate(fpr), true positive rate(tpr), thresholds

fpr_logreg, tpr_logreg, thresholds_logreg = metrics.roc_curve(y_train, y_pred_prob_logreg_class1)
fpr_rfc, tpr_rfc, thresholds_rfc = metrics.roc_curve(y_train, y_pred_prob_rfc_class1)
fpr_SGD, tpr_SGD, thresholds_SGD = metrics.roc_curve(y_train, y_pred_prob_SGD)

plt.plot(fpr_logreg, tpr_logreg, label="logreg")
plt.plot(fpr_rfc, tpr_rfc, label="rfc")
plt.plot(fpr_SGD, tpr_SGD, label="SGD")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True)
y_pred_prob_logreg=clf_logreg.predict(X_test)
y_pred_prob_rfc=clf_rfc.predict(X_test)
y_pred_prob_SGD=clf_SGD.predict(X_test)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100, random_state = 0,learning_rate = 1,max_features=5)
gbc.fit(X_train,y_train)
print('For Gradient Boost Classifier')
train_score = metrics.roc_auc_score(y_train, gbc.predict_proba(X_train)[:,1])
print('Train roc_auc_score:',train_score)
test_score = metrics.roc_auc_score(y_test, gbc.predict_proba(X_test)[:,1])
print("Test roc_auc_score:",test_score)
print(metrics.roc_auc_score(y_train, clf_logreg.predict(X_train)))
print(metrics.roc_auc_score(y_train, clf_rfc.predict(X_train)))
print(metrics.roc_auc_score(y_train, clf_SGD.predict(X_train)))
print('Train roc_auc_score_gradientboost:',train_score)
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(y_test, y_pred_prob_logreg))
print(metrics.roc_auc_score(y_test, y_pred_prob_rfc))
print(metrics.roc_auc_score(y_test, y_pred_prob_SGD))
print("Test roc_auc_score:",test_score)
a=clf_rfc.predict_proba(testset)
b=a[:,1]
ss=ss.drop(['Attrition'],axis=1)
ss['Attrition']=b
ss.to_csv('submission.csv',index=False)
ss.head()
