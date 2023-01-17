# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV



style.use('fivethirtyeight')
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.shape
# Scale the Amount and the time column

from sklearn.preprocessing import StandardScaler, RobustScaler

rb_scaler = RobustScaler()



df['scaled_amt'] = rb_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df['scaled_time'] = rb_scaler.fit_transform(df['Time'].values.reshape(-1, 1))



df.drop(['Amount', 'Time'], axis = 1, inplace = True)





sns.countplot(df['Class'], color = 'green')
# X = df.drop(['Class'], axis = 1)

# y = df['Class']
# Create a balanced dataset with equal number of fraud and non fraud data

# shuffel the data

df = df.sample(frac = 1, random_state = 1)



fraud_df = df.loc[df['Class']==1]

# pick 492 recirds of non fraud data

non_fraud_df = df.loc[df['Class']==0][:492]

normalized_df = pd.concat([fraud_df, non_fraud_df])

normalized_df.shape

# again shuffel

new_df = normalized_df.sample(frac = 1)



sns.countplot(new_df['Class'])
# corelation of features for new normalized, balanced data set

plt.figure(figsize=(18, 12))

sns.heatmap(new_df.corr(), annot=True)
fig, ax = plt.subplots(2, 4, figsize = (18, 10))



sns.boxplot(x= 'Class', y = 'V12', data = new_df, ax = ax[0,0])

ax[0,0].set_title("V12 vs Class Negative Corelation")



sns.boxplot(x= 'Class', y = 'V14', data = new_df, ax = ax[0,1])

ax[0,1].set_title("V14 vs Class Negative Corelation")



sns.boxplot(x= 'Class', y = 'V16', data = new_df, ax = ax[0,2])

ax[0,2].set_title("V16 vs Class Negative Corelation")



sns.boxplot(x= 'Class', y = 'V10', data = new_df, ax = ax[0,3])

ax[0,3].set_title("V10 vs Class Negative Corelation")



sns.boxplot(x= 'Class', y = 'V4', data = new_df, ax = ax[1,0])

ax[1,0].set_title("V4 vs Class Positive Corelation")



sns.boxplot(x= 'Class', y = 'V11', data = new_df, ax = ax[1,1])

ax[1,1].set_title("V11 vs Class Positive Corelation")



sns.boxplot(x= 'Class', y = 'V2', data = new_df, ax = ax[1,2])

ax[1,2].set_title("V2 vs Class Positive Corelation")
# remove outliers 

outline_col = ['V14', 'V16', 'V10']

for col in outline_col:

    class1 = new_df[col].loc[new_df['Class']==1].values

    q25, q75 = np.percentile(class1, 25), np.percentile(class1, 75)

    print('{} - 25 percentile : {} 75 percent : {}'.format(col,q25, q75))

    iqr = q75-q25

    print('{} IQR : {}'.format(col, iqr))

    interquartile_range = iqr*1.5

    interquartile_range_lower,  interquartile_range_higher = q25 - interquartile_range, q75+interquartile_range

    print('{}-Lower Range : {} Higher Range : {}'.format(col,interquartile_range_lower, interquartile_range_higher))

    outliner = [x for x in class1 if(x < interquartile_range_lower or x> interquartile_range_higher)]

#     print(len(outliner))



    #  delete the outliners from the new df

    new_df = new_df.drop(new_df.loc[(new_df[col]< interquartile_range_lower) | 

                                    (new_df[col] > interquartile_range_higher)].index)

    new_df.shape
fig, ax = plt.subplots(1, 3, figsize = (18, 10))

i = 0

for i, col in zip(range(len(outline_col)),outline_col):

    sns.boxplot(x= 'Class', y = col, data = new_df, ax = ax[i])

    ax[i].set_title("{} vs Class After outliners deleted".format(col))
X = new_df.drop(['Class'], axis = 1)

y = new_df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

print('X Train Shape', X_train.shape)

print('Y Train Shape', y_train.shape)

print('X Test Shape', X_test.shape)

print('y Train Shape', y_test.shape)
models = []

models.append(('LSR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(("SVC", SVC()))

models.append(('DTC', DecisionTreeClassifier()))
for name, model in models:

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    score = accuracy_score(y_test, prediction)

    print('{}Accuracy Score : {}%'.format(name, round(score, 4)*100))
estimator = LogisticRegression()

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],

             "penalty": ['l1', 'l2']}



grid_search = GridSearchCV(estimator=estimator, param_grid = param_grid, cv=5,verbose=0)

grid_search.fit(X_train, y_train)

log_reg = grid_search.best_estimator_

print(log_reg)

# Hyperparameters for KNN

estimator = KNeighborsClassifier()

param_grid = {'n_neighbors' : [5, 7, 10,12, 15, 20]}



grid_search = GridSearchCV(estimator=estimator, param_grid = param_grid, cv=5,verbose=0)

grid_search.fit(X_train, y_train)

knn_clf = grid_search.best_estimator_

print(knn_clf)
estimator = SVC()

param_grid = {'kernel' : ['linear', 'rbf', 'poly'],

             'gamma' : [0.1, 1, 10, 100],

             'C' : [0.1, 1, 10, 100, 1000]}



grid_svc = GridSearchCV(estimator=estimator, param_grid = param_grid, cv=5,verbose=0)

grid_svc.fit(X_train, y_train)

svc_best = grid_svc.best_estimator_

print(svc_best)
estimator = DecisionTreeClassifier()

param_grid = {'max_depth' : [2,3,4,5],

             'min_samples_split' : [2,3,4]}



grid_dtc = GridSearchCV(estimator=estimator, param_grid = param_grid, cv=5,verbose=0)

grid_dtc.fit(X_train,y_train)

dtc_best = grid_dtc.best_estimator_

print(dtc_best)
from sklearn.model_selection import cross_val_score

cv_score_hyper_params_logReg = cross_val_score(log_reg, X, y, cv = 5)

print('Logestic Regression Accuracy Score : {}%'.format(round(cv_score_hyper_params_logReg.mean(), 4)*100))
cv_score_hyper_params_logReg = cross_val_score(knn_clf, X, y, cv = 5)

print('KNN Accuracy Score : {}%'.format(round(cv_score_hyper_params_logReg.mean(), 4)*100))