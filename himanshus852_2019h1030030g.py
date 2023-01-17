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

from matplotlib import pyplot as plt

import pandas as pd

from scipy import optimize

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier

import imblearn

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB 
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv",header=0)

df
df.groupby('target').mean()
df.head()
df.info()
df.describe()
from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler

from imblearn.pipeline import Pipeline

from imblearn.over_sampling import RandomOverSampler

from sklearn import preprocessing
#overSample = SMOTE()

#underSample = RandomUnderSampler()
#overSample = RandomOverSampler(sampling_strategy=0.3)

underSample = RandomUnderSampler(sampling_strategy='majority')
# steps = [('o', overSample), ('u', underSample)]

# pipeline = Pipeline(steps=steps)
y = df['target']

X = df.drop(["id","target"],axis =1)
count = 0

for x in y:

    if x==0:

        count+=1

    

print(count)

print(len(y))

#X, y = pipeline.fit_resample(X, y)

#X, y = overSample.fit_resample(X, y)
# count = 0

# for x in y:

#     if x==0:

#         count+=1

    

# print(count)

# print(len(y))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=121)
count = 0

for val in y_train:

    if val==0:

        count+=1

    

print(count)

print(len(y_train))
#X_train, y_train = overSample.fit_resample(X_train, y_train)

#X_train, y_train = pipeline.fit_resample(X_train, y_train)

X_train, y_train = underSample.fit_resample(X_train, y_train)
count = 0

for val in y_train:

    if val==0:

        count+=1

    

print(count)

print(len(y_train))

print(len(X_train))
#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)

#X_test = scaler.fit_transform(X_test)

# X_scaled = scaler.fit_transform(X)

# y_scaled = scaler.fit_transform(y)
#X_train, y_train = overSample.fit_resample(X_train, y_train)
X_train
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB 
log_model = LogisticRegression(max_iter=4000)

#tree_model = DecisionTreeClassifier()

#xgb_model = XGBClassifier()

#rf_model = RandomForestClassifier(n_estimators=17 , max_samples=0.39)

#gaussian = GaussianNB()
log_model.fit(X_train,y_train)

#tree_model.fit(X_train,y_train)

#xgb_model.fit(X_train,y_train)

#rf_model.fit(X_train,y_train)

#gaussian.fit(X_train,y_train)
y_pred = log_model.predict(X_test)

#y_pred = tree_model.predict(X_test)

#y_pred = xgb_model.predict(X_test)

#y_pred = rf_model.predict(X_test)

#y_pred = gaussian.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#y_pred_log = log_model.decision_function(X_test)
#from sklearn.metrics import roc_curve , auc


#log_fpr,log_tpr,threshold = roc_curve(y_test,y_pred_log)
#auc_log = auc(log_fpr,log_tpr)
#plt.figure(figsize=(5,5),dpi=100)
# plt.plot(log_fpr,log_tpr,marker=".",label="Logistic (auc = %0.3f)"% auc_log)

# plt.xlabel("FPR")

# plt.ylabel("TPR")

# plt.legend()

# plt.show()
df_test_import = pd.read_csv("/kaggle/input/minor-project-2020/test.csv",header=0)
df_test = df_test_import.drop("id",axis=1)
df_test
#df_test = scaler.fit_transform(df_test)
df_test_pred = log_model.predict(df_test)

#df_test_pred = tree_model.predict(df_test)

#df_test_pred = xgb_model.predict(df_test)

#df_test_pred = rf_model.predict(df_test)

#df_test_pred = gaussian.predict(df_test)
df_test_pred
count = 0

for x in df_test_pred:

    if x==0:

        count+=1

    

print(count)

print(len(df_test_pred))
df_test_import.id
submission = pd.DataFrame({"id":df_test_import.id , "target":df_test_pred})
submission.to_csv('Approach_9_submission_1.csv',index=False)
submission.head()