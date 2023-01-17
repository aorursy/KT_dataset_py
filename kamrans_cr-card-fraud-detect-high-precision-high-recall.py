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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV

from imblearn.under_sampling import NearMiss

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_score, f1_score

from imblearn.over_sampling import RandomOverSampler

from collections import Counter
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
df.info()
df.describe().T
plt.figure(figsize=(12, 6))

sns.countplot(df['Class'])

print('Count 0s = {}'.format((df['Class']==0).sum()))

print('Count 1s = {}'.format((df['Class']==1).sum()))

print('Percentage of fraud transactions = {}%'.format((df['Class']==1).sum() / (df['Class']==0).sum() * 100))
# Each feature visualized



for each in df.drop(columns='Class').columns:

    print('Please scroll down this winndow')

    fig = plt.subplots(figsize=(12, 4))

    sns.distplot(df[each], kde=False)

    plt.show()
# Each feature visualised by class



df0s = df[df['Class']==0]

df1s = df[df['Class']==1]



for each in df.drop(columns='Class').columns:

    print('Please scroll down this window')

    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(12, 2))

    sns.distplot(df0s[each], kde=False, ax=axs[0], label='Class = 0')

    axs[0].legend()

    sns.distplot(df1s[each], kde=False, ax=axs[1], label='Class = 1')

    axs[1].legend()

    plt.show()
X = df.drop(columns='Class').values

y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
nm = NearMiss()

X_train_us, y_train_us = nm.fit_resample(X_train, y_train)

print('Shape of X_train is {}, and y_train is {}'.format(X_train.shape, y_train.shape))

print('Shape of UNDERSAMPLED X_train_us is {}, and y_train_us is {}'.format(X_train_us.shape, y_train_us.shape))
clf_models = []

clf_models.append(('MLPClf', MLPClassifier()))

clf_models.append(('KNN', KNeighborsClassifier()))

clf_models.append(('SVC', SVC()))

clf_models.append(('GausProcess', GaussianProcessClassifier()))

clf_models.append(('DTree', DecisionTreeClassifier()))

clf_models.append(('RandForest', RandomForestClassifier()))

clf_models.append(('AdaBoost', AdaBoostClassifier()))

clf_models.append(('GausNB', GaussianNB()))

clf_models.append(('QuadDiscAna', QuadraticDiscriminantAnalysis()))



results = []

names = []



for name, model in clf_models:

    kfold = KFold(n_splits=5, shuffle=True)

    cv_results = cross_val_score(model, X_train_us, y_train_us, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    note = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(note)
plt.figure(figsize=(15, 8))

sns.boxplot(names, results)
abc = AdaBoostClassifier()



abc_param = {'n_estimators' : [140], 'learning_rate' : [0.5]}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



abc_gs = GridSearchCV(abc, param_grid=abc_param, cv=5)



abc_gs.fit(X_train_us, y_train_us)
print(abc_gs.best_params_)
adaboost = AdaBoostClassifier(**abc_gs.best_params_)



adaboost.fit(X_train_us, y_train_us)
ada_preds = adaboost.predict(X_test)
ada_cm = confusion_matrix(y_test, ada_preds)



sns.heatmap(ada_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
ada_cr = classification_report(y_test, ada_preds)

print('AdaBoostClassifier classification report' '\n', ada_cr)
ada_us_metrics = {'Precision' : precision_score(y_test, ada_preds), 'Recall' : recall_score(y_test, ada_preds),

                  'F1 Score' : f1_score(y_test, ada_preds)}

print(ada_us_metrics)
rf = RandomForestClassifier()



rf_param = {'n_estimators' : [50, 60, 70], 'max_depth' : [None, 30, 50], 

            'min_samples_split' : [40, 45, 50, 55, 60], 'min_samples_leaf' : [1], 

            'max_features' : ['log2']}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



rf_gs = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5, n_jobs=-2)



rf_gs.fit(X_train_us, y_train_us)
rf_gs.best_params_
rf_clf = RandomForestClassifier(**rf_gs.best_params_)



rf_clf.fit(X_train_us, y_train_us)
rf_preds = rf_clf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_preds)



sns.heatmap(rf_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
rf_cr = classification_report(y_test, rf_preds)

print('Random Forest Classifier classificaction report' '\n', rf_cr)
rf_us_metrics = {'Precision' : precision_score(y_test, rf_preds), 'Recall' : recall_score(y_test, rf_preds),

                  'F1 Score' : f1_score(y_test, rf_preds)}

print(rf_us_metrics)
dtree = DecisionTreeClassifier()



dtree_grid = {'max_depth' : [None, 6, 7], 'min_samples_split' : [6, 7], 'min_samples_leaf' : [1, 2]}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



dtree_gs = GridSearchCV(estimator=dtree, param_grid=dtree_grid, cv=5)



dtree_gs.fit(X_train_us, y_train_us)
dtree_gs.best_params_
dtree_clf = DecisionTreeClassifier(**dtree_gs.best_params_)

dtree_clf.fit(X_train_us, y_train_us)
dtree_preds = dtree_clf.predict(X_test)
dtree_cm = confusion_matrix(y_test, dtree_preds)



sns.heatmap(dtree_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
dtree_cr = classification_report(y_test, dtree_preds)

print('Decision Tree Classifier classificaction report' '\n', rf_cr)
dtree_us_metrics = {'Precision' : precision_score(y_test, dtree_preds), 'Recall' : recall_score(y_test, dtree_preds),

                  'F1 Score' : f1_score(y_test, dtree_preds)}

print(dtree_us_metrics)
# Over sampling the training set only



ros = RandomOverSampler()

X_train_os, y_train_os = ros.fit_sample(X_train, y_train)
print('Original training dataset class counts {}'.format(Counter(y_train)))

print('Over sampled dataset class counts {}'.format(Counter(y_train_os)))
# abc = AdaBoostClassifier()



# abc_param = {'n_estimators' : [2000], 'learning_rate' : [1.0]}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



# abc_gs = GridSearchCV(abc, param_grid=abc_param, cv=5)



# abc_gs.fit(X_train_os, y_train_os)
# print(abc_gs.best_params_)
adaboost = AdaBoostClassifier(n_estimators=2000, learning_rate=1.0)



adaboost.fit(X_train_os, y_train_os)
ada_preds = adaboost.predict(X_test)
ada_cm = confusion_matrix(y_test, ada_preds)



sns.heatmap(ada_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
ada_cr = classification_report(y_test, ada_preds)

print('AdaBoostClassifier classification report' '\n', ada_cr)
ada_os_metrics = {'Precision' : precision_score(y_test, ada_preds), 'Recall' : recall_score(y_test, ada_preds),

                  'F1 Score' : f1_score(y_test, ada_preds)}

print(ada_os_metrics)
rf = RandomForestClassifier()



rf_param = {'n_estimators' : [200, 300, 400, 500, 1000, 2000, 3000], 'max_depth' : [30], 

            'min_samples_split' : [30], 'max_features' : ['log2']}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



rf_gs = GridSearchCV(estimator=rf, param_grid=rf_param, cv=5, n_jobs=-2)



rf_gs.fit(X_train_os, y_train_os)
rf_gs.best_params_
rf_clf = RandomForestClassifier(**rf_gs.best_params_)



rf_clf.fit(X_train_os, y_train_os)
rf_preds = rf_clf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_preds)



sns.heatmap(rf_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
rf_cr = classification_report(y_test, rf_preds)

print('Random Forest Classifier classificaction report' '\n', rf_cr)
rf_os_metrics = {'Precision' : precision_score(y_test, rf_preds), 'Recall' : recall_score(y_test, rf_preds),

                  'F1 Score' : f1_score(y_test, rf_preds)}

print(rf_os_metrics)
dtree = DecisionTreeClassifier()



dtree_grid = {'splitter' : ['best'], 'max_depth' : [None],

              'min_samples_split' : [7], 'min_samples_leaf' : [1]}

# Why am I using this parameter grid with these particular values? 

# I have run grid search several times on this dataset before, 

# in order to save resources and time, I am using the relevant values only



dtree_gs = GridSearchCV(estimator=dtree, param_grid=dtree_grid, cv=5)



dtree_gs.fit(X_train_os, y_train_os)
dtree_gs.best_params_
dtree_clf = DecisionTreeClassifier(**dtree_gs.best_params_)

dtree_clf.fit(X_train_os, y_train_os)
dtree_preds = dtree_clf.predict(X_test)
dtree_cm = confusion_matrix(y_test, dtree_preds)



sns.heatmap(dtree_cm, annot=True, annot_kws={'size' : 24}, fmt='g')

plt.xlabel('Predictions', size=24)

plt.ylabel('y-true', size=24)
dtree_cr = classification_report(y_test, dtree_preds)

print('Decision Tree Classifier classificaction report' '\n', rf_cr)
dtree_os_metrics = {'Precision' : precision_score(y_test, dtree_preds), 'Recall' : recall_score(y_test, dtree_preds),

                  'F1 Score' : f1_score(y_test, dtree_preds)}

print(ada_os_metrics)
all_metrics = [ada_us_metrics, rf_us_metrics, dtree_us_metrics, ada_os_metrics, rf_os_metrics, ada_os_metrics]

print(all_metrics)
mdf = pd.DataFrame(all_metrics, index= ['ada_under_samp', 'rf_under_samp', 'dtree_undersamp', 

                                        'ada_over_samp', 'rf_over_samp', 'ada_over_samp'])

mdf
mdf.plot(kind='bar', figsize=(12, 8))