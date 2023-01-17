import numpy as np 

import pandas as pd 

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



from sklearn.datasets import load_breast_cancer

from sklearn.datasets import load_digits

from sklearn import metrics

%matplotlib inline



import os

import warnings

warnings.filterwarnings('ignore')

cancer = load_breast_cancer()

digits = load_digits()



data = cancer
data
df = pd.DataFrame(data= np.c_[data['data'], data['target']],

                     columns= list(data['feature_names']) + ['target'])

df['target'] = df['target'].astype('uint16')
df
df.head()
import pandas_profiling as pp

profile = pp.ProfileReport(df) 

profile.to_file("EDA.html")


# create x and y train

X = df.drop('target', axis=1)

y = df[['target']]



# split data into train and test/validation sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# check the average cancer occurence rates in train and test data, should be comparable

y_train.mean(),y_test.mean()
X_train.info()
from catboost import CatBoostClassifier

model=CatBoostClassifier()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

categorical_features_indices
model.fit(X_train,y_train,cat_features=([]))



# test error

y_pred = model.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

score
import sklearn.metrics



sklearn.metrics.accuracy_score(y_test,model.predict(X_test))
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test,model.predict(X_test))

sklearn.metrics.auc(fpr, tpr)
sklearn.metrics.average_precision_score(y_test,model.predict(X_test))
sklearn.metrics.balanced_accuracy_score(y_test,model.predict(X_test))
sklearn.metrics.brier_score_loss(y_test,model.predict(X_test))
print(sklearn.metrics.classification_report(y_test,model.predict(X_test)))

sklearn.metrics.cohen_kappa_score(y_test,model.predict(X_test))
print(sklearn.metrics.confusion_matrix(y_test,model.predict(X_test)))
sklearn.metrics.f1_score(y_test,model.predict(X_test))
sklearn.metrics.fbeta_score(y_test,model.predict(X_test),beta=.5)
sklearn.metrics.hamming_loss(y_test,model.predict(X_test))
sklearn.metrics.hinge_loss(y_test,model.predict(X_test))
sklearn.metrics.jaccard_score(y_test,model.predict(X_test))
sklearn.metrics.log_loss(y_test,model.predict(X_test))
sklearn.metrics.matthews_corrcoef(y_test,model.predict(X_test))
sklearn.metrics.precision_recall_curve(y_test,model.predict(X_test))
sklearn.metrics.precision_recall_fscore_support(y_test,model.predict(X_test))
sklearn.metrics.precision_score(y_test,model.predict(X_test))
sklearn.metrics.recall_score(y_test,model.predict(X_test))
sklearn.metrics.roc_auc_score(y_test,model.predict(X_test))
sklearn.metrics.roc_curve(y_test,model.predict(X_test))
sklearn.metrics.zero_one_loss(y_test,model.predict(X_test))