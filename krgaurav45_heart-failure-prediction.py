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
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.head()
df.info()
df['DEATH_EVENT'].value_counts()
df.describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



def univariate(df,col):

    fig = plt.figure()

    sns.set_style('darkgrid')

    ax = sns.FacetGrid(df,hue='DEATH_EVENT', height=5,aspect=2.0)

    ax.map(sns.distplot, col).add_legend()

    plt.title('distribution of {}'.format(col))

    plt.show()

    

def cat_plot(df,col):

    sns.set_style('darkgrid')

    fig = plt.figure(figsize=(12,5))

    sns.countplot(data=df,x = col, hue = 'DEATH_EVENT')

    plt.title('distribution of {}'.format(col))

    plt.show()



univariate(df,'age')
univariate(df, 'creatinine_phosphokinase')
univariate(df, 'ejection_fraction')

univariate(df,'platelets')
univariate(df,'serum_creatinine')

univariate(df, 'serum_sodium')
univariate(df, 'time')
cat_plot(df,'anaemia')

cat_plot(df,'diabetes')
cat_plot(df,'high_blood_pressure')

cat_plot(df,'sex')

cat_plot(df,'smoking')
cat_col = []

num_col = []

for c in df.columns:

    if len(df[c].unique()) != 2:

        num_col.append(c)

    else:

        cat_col.append(c)

        

print(num_col)

print(cat_col)
sns.pairplot(df[num_col+['DEATH_EVENT']], hue="DEATH_EVENT")
fig = plt.figure()

sns.set_style('darkgrid')

ax = sns.FacetGrid(df,hue='DEATH_EVENT', height=5,aspect=2.0)

ax.map(sns.scatterplot, 'platelets', 'age').add_legend()

plt.title('distribution of time and age')

plt.show()
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, roc_curve

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import Normalizer

from scipy.stats import randint as sp_randint

from xgboost import XGBClassifier

import warnings

warnings.filterwarnings('ignore')
y = df['DEATH_EVENT']

X = df.drop(columns=['DEATH_EVENT'],axis=1)



X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.1, random_state=42, stratify=y)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
for c in num_col:

    nr = Normalizer()

    nr.fit(X_train[c].values.reshape(1,-1))

    X_train[c] = nr.transform(X_train[c].values.reshape(1,-1)).T

    X_test[c] = nr.transform(X_test[c].values.reshape(1,-1)).T
alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

auc1 = []

auc2 = []

for i in alpha:

    sgd = SGDClassifier(loss='log',alpha=i, n_jobs=-1, random_state=42)

    sgd.fit(X_train,y_train)

    y_scores = sgd.predict_proba(X_train)[:,1]

    y_test_scores = sgd.predict_proba(X_test)[:,1] 

    

    auc1.append(roc_auc_score(y_train,y_scores))

    auc2.append(roc_auc_score(y_test,y_test_scores))

    

plt.plot(np.log(alpha),auc1,label='train auc')

plt.plot(np.log(alpha),auc2,label='test auc')

plt.legend()

plt.show()
sgd = SGDClassifier(n_jobs=-1, random_state=42, loss='log')

params = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

clf = RandomizedSearchCV(sgd,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)
def plot_roc(clf):

    try:

        y_scores = clf.predict_proba(X_train)[:,1]

        y_test_scores = clf.predict_proba(X_test)[:,1]

    except:

        y_scores = clf.decision_function(X_train)

        y_test_scores = clf.decision_function(X_test)

    fpr,tpr,th = roc_curve(y_train,y_scores)

    fpr1,tpr1,th1 = roc_curve(y_test,y_test_scores)



    plt.plot(fpr,tpr,label='train')

    plt.plot(fpr1,tpr1,label='test')

    plt.legend()

    plt.show()

    

lr = clf.best_estimator_

lr.fit(X_train,y_train)

plot_roc(lr)
def metrics(clf):

    try:

        y_scores = clf.predict_proba(X_train)[:,1]

        y_test_scores = clf.predict_proba(X_test)[:,1]

    except:

        y_scores = clf.decision_function(X_train)

        y_test_scores = clf.decision_function(X_test)

    print("train AUC:", roc_auc_score(y_train,y_scores))

    print("test AUC:", roc_auc_score(y_test,y_test_scores))

    y_pred = clf.predict(X_train)

    y_test_pred = clf.predict(X_test)

    print("train f1:", f1_score(y_train,y_pred))

    print("test f1:", f1_score(y_test,y_test_pred))

    

    fig,ax = plt.subplots(1,2,figsize=(16,4))

    ax1 = sns.heatmap(confusion_matrix(y_train,y_pred), annot=True, ax=ax[0])

    ax1.set_title("train confusion matrix")

    ax1.set_xlabel("predicted")

    ax1.set_ylabel("original")

    ax2 = sns.heatmap(confusion_matrix(y_test,y_test_pred), annot=True, ax=ax[1])

    ax2.set_title("test confusion matrix")

    ax2.set_xlabel("predicted")

    ax2.set_ylabel("original")

    plt.show()

    

metrics(lr)
sgd = SGDClassifier(loss='hinge', n_jobs=-1, random_state=42)

params = {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}

clf = RandomizedSearchCV(sgd,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)
svm = clf.best_estimator_

svm.fit(X_train,y_train)

plot_roc(svm)

metrics(svm)
np.random.seed(42)

dt = DecisionTreeClassifier(random_state=42)

params = {'max_depth':sp_randint(3,500), 'min_samples_split':sp_randint(2,250),

          'min_samples_leaf':sp_randint(2,50)

         }

clf = RandomizedSearchCV(dt,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)
dt = clf.best_estimator_

dt.fit(X_train,y_train)

plot_roc(dt)

metrics(dt)
np.random.seed(42)

rf = RandomForestClassifier(n_jobs=-1,random_state=42)

params = {'n_estimators':sp_randint(1,500), 'max_depth':sp_randint(3,100), 'min_samples_split':sp_randint(2,250),

          'min_samples_leaf':sp_randint(2,50)

         }

clf = RandomizedSearchCV(rf,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)



rf = clf.best_estimator_

rf.fit(X_train,y_train)

plot_roc(rf)

metrics(rf)
def imp_plot(clf):

    features = X_train.columns

    importance = clf.feature_importances_

    indices = np.argsort(importance)

    plt.figure(figsize=(12,6))

    plt.barh(range(len(features)),importance[indices])

    plt.yticks(range(len(features)),[features[i] for i in indices],rotation=45)

    plt.title("Important features")

    plt.show()

    

imp_plot(rf)
np.random.seed(42)

xgb = XGBClassifier(n_jobs=-1,random_state=42)

params = {'learning_rate':alpha, 'n_estimators':sp_randint(1,500), 'max_depth':sp_randint(3,100), 

          'colsample_bytree':[0.1,0.2,0.4,0.5,0.7,1], 'subsamples':[0.1,0.3,0.5,0.8,1]}



clf = RandomizedSearchCV(xgb,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)



xgb = clf.best_estimator_

xgb.fit(X_train,y_train)

plot_roc(xgb)

metrics(xgb)

imp_plot(xgb)
np.random.seed(42)

xgb = AdaBoostClassifier(random_state=42)

params = {'learning_rate':alpha, 'n_estimators':sp_randint(1,500)}



clf = RandomizedSearchCV(xgb,params,n_jobs=-1,random_state=42, return_train_score=True,scoring='roc_auc',cv=10)

clf.fit(X_train,y_train)

print(clf.best_estimator_,clf.best_params_,clf.best_score_)



xgb = clf.best_estimator_

xgb.fit(X_train,y_train)

plot_roc(xgb)

metrics(xgb)

imp_plot(xgb)