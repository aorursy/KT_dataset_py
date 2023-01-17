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
import numpy as np
import pandas as pd
import random
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV 

from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost  import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

%matplotlib inline
sns.set(color_codes=True)
df_train=pd.read_csv("/kaggle/input/novartis-challenge-hacking-prediction/Train.csv")
df_train.shape
df_test=pd.read_csv("/kaggle/input/novartis-challenge-hacking-prediction/Test.csv")
df_test.shape
f,ax=plt.subplots(1,2,figsize=(16,6))
labels = ['Hacked', 'Not Hacked']
df_train['MULTIPLE_OFFENSE'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True,labels=labels,fontsize=10)
sns.countplot('MULTIPLE_OFFENSE',data=df_train, ax=ax[1])
ax[1].set_xticklabels(['Hacked', 'Not Hacked'], fontsize=10)
plt.show()
df_train['MULTIPLE_OFFENSE'].value_counts()
df_train.head(5)
df_test.head(5)
df_train.isnull().sum()
df_test.isnull().sum()
df_train.info()
df_test.info()
df_train['X_12'].fillna(df_train['X_12'].mode()[0], inplace=True) # Replace empty/null values with "NaN"
df_test['X_12'].fillna(df_test['X_12'].mode()[0], inplace=True) # Replace empty/null values with "NaN"

df_train.describe().T
plt.subplots(figsize=(15, 6))
sns.boxplot(data = df_train, orient = 'v')
df_train['X_12'] = df_train['X_12'].astype('int64') 
df_test['X_12'] = df_test['X_12'].astype('int64') 
df_train['MULTIPLE_OFFENSE'] = df_train['MULTIPLE_OFFENSE'].astype('category') 
df_train.hist(bins=10, figsize=(14,10))
plt.show()
f, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(df_train.corr(method='spearman'), annot=True, cmap="YlGnBu")
#X_2 and X_3 are highly correlated.
sns.jointplot(df_train['X_2'],df_train['X_3'], kind="reg", color="g")
#Drop Duplicate rows

df_train = df_train.drop(['INCIDENT_ID', 'DATE'], axis=1)

df_train.drop_duplicates(keep='first', inplace=True)

df_train.shape
X = df_train.drop(['MULTIPLE_OFFENSE'], axis=1)
y = df_train['MULTIPLE_OFFENSE']
# define oversampling strategy
from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
print(X.shape, ' ', y.shape)
print(X_over.shape, ' ', y_over.shape)
X_train, X_val, y_train, y_val = train_test_split(X_over, y_over ,test_size=0.3, random_state=10)
print(X_train.shape, ' ', y_train.shape)
print(X_val.shape,   ' ', y_val.shape)
#Use standardscaler to standardize the features

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_val_sc   = sc.transform(X_val)
#Use standardscaler to standardize the features
#X_test = df_test.drop(['INCIDENT_ID','DATE','X_2'], axis=1)
X_test = df_test.drop(['INCIDENT_ID','DATE'], axis=1)
X_test_sc = sc.transform(X_test)
#CATBoost

from catboost import CatBoostClassifier
cb_clf = CatBoostClassifier(learning_rate=0.1, n_estimators=1000, subsample=0.70, max_depth=5, scale_pos_weight=2.5, silent=True)
cb_clf.fit(X_train_sc, y_train)

# evaluate predictions
y_train_predict_cb = cb_clf.predict(X_train_sc)
print('Train Accuracy %.3f' % metrics.accuracy_score(y_train, y_train_predict_cb))

# make predictions for test data
y_pred_cb = cb_clf.predict(X_val_sc)
predictions = [round(value) for value in y_pred_cb]

print('Test Accuracy %.3f' % metrics.accuracy_score(y_val, predictions))
print(metrics.confusion_matrix(y_val, predictions))
print(metrics.classification_report(y_val, predictions))
print('Precision Score %.3f' % metrics.precision_score(y_val, predictions))
print('Recall Score %.3f' % metrics.recall_score(y_val, predictions))
print('F1 Score %.3f' % metrics.f1_score(y_val, predictions))

#XGBOOST

from xgboost import XGBClassifier

xg = XGBClassifier(scale_pos_weight=2.5 ,silent=True)
    
xg.fit(X_train_sc, y_train)


# evaluate predictions
y_train_predict_xg = xg.predict(X_train_sc)
print('Train Accuracy %.3f' % metrics.accuracy_score(y_train, y_train_predict_xg))

# make predictions for test data
y_pred_xg = xg.predict(X_val_sc)
predictions = [round(value) for value in y_pred_xg]


#y_predict = xg.predict(X_val_sc)
#redictProb_xg = xg.predict_proba(X_val_sc)

print('Test Accuracy %.3f' % metrics.accuracy_score(y_val, predictions))
print(metrics.confusion_matrix(y_val, predictions))
print(metrics.classification_report(y_val, predictions))
print('Precision Score %.3f' % metrics.precision_score(y_val, predictions))
print('Recall Score %.3f' % metrics.recall_score(y_val, predictions))
print('F1 Score %.3f' % metrics.f1_score(y_val, predictions))

#Voting classifier V2
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection



clf1 = XGBClassifier(n_estimators=900, scale_pos_weight = 2.5, silent=True)

clf2 = CatBoostClassifier(silent=True, learning_rate=0.1, n_estimators=900, scale_pos_weight=2.5)
                                                                                    
clf6   = VotingClassifier(estimators=[('XGB', clf1),('CB', clf2)],  voting='hard')

scores = model_selection.cross_val_score(clf6, X_train_sc, y_train, cv=3, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) " % (scores.mean(), scores.std() ))

clf6.fit(X_train_sc, y_train)

# evaluate predictions
y_train_predict_vot = clf6.predict(X_train_sc)
print('Train Accuracy %.3f' % metrics.accuracy_score(y_train, y_train_predict_vot))

# make predictions for test data
y_pred_clf6 = clf6.predict(X_val_sc)
predictions = [round(value) for value in y_pred_clf6]

print('Test Accuracy %.3f' % metrics.accuracy_score(y_val, predictions))
print(metrics.confusion_matrix(y_val, predictions))
print(metrics.classification_report(y_val, predictions))
print('Precision Score %.3f' % metrics.precision_score(y_val, predictions))
print('Recall Score %.3f' % metrics.recall_score(y_val, predictions))
print('F1 Score %.3f' % metrics.f1_score(y_val, predictions))


#Recall 99.68 during evaluation
predict_test_voting = clf6.predict(X_test_sc)
submission_df = pd.DataFrame({'INCIDENT_ID':df_test['INCIDENT_ID'], 'MULTIPLE_OFFENSE':predict_test_voting})
submission_df.to_csv('Submission_VOT_v2.csv', index=False)