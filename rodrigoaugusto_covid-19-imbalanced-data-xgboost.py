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

from scipy import stats

from scipy.stats import spearmanr

from scipy.stats import kendalltau

import missingno as msno

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["figure.figsize"] = (15,8)
excel_file = pd.ExcelFile('/kaggle/input/covid19/dataset.xlsx')

dfs = {sheet_name: excel_file.parse(sheet_name) 

          for sheet_name in excel_file.sheet_names}



df_covid = dfs['All']



df_covid.head()
print('Proportion between results:\n',(df_covid['SARS-Cov-2 exam result'].value_counts()/len(df_covid.index))*100)
print('Remove columns having more than 76.045358% of missing data')

pd.set_option('display.max_columns', 500)

total_cases = len(df_covid.index)

dic_nan = {}

for item in df_covid.columns:

    dic_nan[item] = (df_covid[item].isnull().values.sum()/total_cases)*100

df_nan = pd.DataFrame.from_dict(dic_nan, orient='index')



df_nan.sort_values(by=0).tail(88).T
# - Drop columns

df_covid_filtered = df_covid[['Patient ID','Patient age quantile','SARS-Cov-2 exam result','Influenza B','Respiratory Syncytial Virus','Influenza A','Rhinovirus/Enterovirus','Inf A H1N1 2009','CoronavirusOC43','Coronavirus229E','Parainfluenza 4','Adenovirus','Chlamydophila pneumoniae','Parainfluenza 3','Coronavirus HKU1','CoronavirusNL63','Parainfluenza 1','Bordetella pertussis','Parainfluenza 2','Metapneumovirus']]



print('Drop rows that still have missing data')

df_covid_filtered.dropna(inplace=True)



print('New ratio of results:\n',(df_covid_filtered['SARS-Cov-2 exam result'].value_counts()/len(df_covid_filtered.index))*100)



# - Map integers for the model

dic_map = {'not_detected': 0, 'detected': 1}

for item in df_covid_filtered.columns:

    try:

         df_covid_filtered = df_covid_filtered.replace({item: dic_map})

    except:

        pass

    

dic_map = {'negative': 0, 'positive': 1}

df_covid_filtered = df_covid_filtered.replace({'SARS-Cov-2 exam result': dic_map})



df_covid_filtered.set_index('Patient ID',inplace=True)



for item in df_covid_filtered.columns:

    df_covid_filtered[item] = df_covid_filtered[item].astype(int)
df_covid_filtered.head()
df_covid_filtered.info()
print('Remove imbalanced features:\n')

total_cases_filtered = len(df_covid_filtered.index)

for item in df_covid_filtered.columns:

    if item == 'Patient ID':

        pass

    else:

        print(item,':')

        print(((df_covid_filtered[item].value_counts())/total_cases_filtered)*100)

        print('\n')
df_covid_filtered.drop(['Parainfluenza 2','Metapneumovirus','Bordetella pertussis','Parainfluenza 1','Coronavirus HKU1','Chlamydophila pneumoniae','Adenovirus','Parainfluenza 4','Coronavirus229E','CoronavirusOC43','Influenza A'],axis=1,inplace=True)
df_covid_filtered.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



from sklearn.metrics import confusion_matrix, plot_confusion_matrix

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



from sklearn.model_selection import KFold 

from sklearn.model_selection import RepeatedKFold

import numpy as np



X = df_covid_filtered.drop('SARS-Cov-2 exam result',axis=1)

Y = df_covid_filtered[['SARS-Cov-2 exam result']]
seed = 7

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = XGBClassifier()

model.fit(X_train, y_train.values.ravel())



y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
y_pred_completo = model.predict(X)

pred = [round(value) for value in y_pred_completo]



y_test_completo = Y.values.ravel()



pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(pred)):

    if pred[x] == y_test_completo[x]:

        if y_test_completo[x] == 1:

            pos_correct += 1

        else:

            neg_correct += 1

       

    if y_test_completo[x] == 1:

        pos_cnt += 1

    else:

        neg_cnt += 1

        

accuracy = accuracy_score(Y, pred)  

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print("pos_acc", pos_correct/pos_cnt*100, "%", '\tAcertos:', pos_correct, ' de ', pos_cnt)

print("neg_acc", neg_correct/neg_cnt*100, "%", '\tAcertos:', neg_correct, ' de ', neg_cnt)
ratio = y_train['SARS-Cov-2 exam result'].value_counts()[0]/y_train['SARS-Cov-2 exam result'].value_counts()[1]

class_weights = {0:1, 1:ratio}

w_array = np.ones(y_train.shape[0], dtype = 'float')



output = y_train.values.ravel()





for i in range(len(output)):

    w_array[i] = class_weights[output[i]]

    

model = XGBClassifier()

model.fit(X_train, y_train.values.ravel(), sample_weight=w_array)
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
plt.rcParams["figure.figsize"] = (7,7)

plot_confusion_matrix(model, X_test, y_test.values.ravel(), cmap = plt.cm.Blues, values_format = '.10g', display_labels = ['negative', 'positive'])
plt.rcParams["figure.figsize"] = (7,7)

plot_confusion_matrix(model, X, Y.values.ravel(), cmap = plt.cm.Blues, values_format = '.10g', display_labels = ['negative', 'positive'])
# grid search positive class weights with xgboost for imbalance classification

from numpy import mean

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier





# define model

model = XGBClassifier()

# define grid

weights = list(range(1,12))

param_grid = dict(scale_pos_weight=weights)

# define evaluation procedure

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid search

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')

# execute the grid search

grid_result = grid.fit(X_train, y_train.values.ravel())



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
y_pred_completo = grid.predict(X_test)

pred = [round(value) for value in y_pred_completo]



y_test_completo = y_test.values.ravel()



pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(pred)):

  if pred[x] == y_test_completo[x]:

    if y_test_completo[x] == 1:

      pos_correct += 1

    else:

      neg_correct += 1

       

  if y_test_completo[x] == 1:

    pos_cnt += 1

  else:

    neg_cnt += 1



accuracy = accuracy_score(y_test, pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print("pos_acc", pos_correct/pos_cnt*100, "%", '\tAcertos:', pos_correct, ' de ', pos_cnt)

print("neg_acc", neg_correct/neg_cnt*100, "%", '\tAcertos:', neg_correct, ' de ', neg_cnt)
y_pred_completo = grid.predict(X)

pred = [round(value) for value in y_pred_completo]



y_test_completo = Y.values.ravel()



pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(pred)):

  if pred[x] == y_test_completo[x]:

    if y_test_completo[x] == 1:

      pos_correct += 1

    else:

      neg_correct += 1

       

  if y_test_completo[x] == 1:

    pos_cnt += 1

  else:

    neg_cnt += 1



accuracy = accuracy_score(Y, pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print("pos_acc", pos_correct/pos_cnt*100, "%", '\tAcertos:', pos_correct, ' de ', pos_cnt)

print("neg_acc", neg_correct/neg_cnt*100, "%", '\tAcertos:', neg_correct, ' de ', neg_cnt)
plot_confusion_matrix(grid, X, Y.values.ravel(), cmap = plt.cm.Blues, values_format = '.10g', display_labels = ['negative', 'positive'])
# grid search positive class weights with xgboost for imbalance classification

from numpy import mean

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier





# define model

model = XGBClassifier()

# define grid

weights = list(range(1,12))

param_grid = dict(scale_pos_weight=weights)

# define evaluation procedure

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid search

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')

# execute the grid search

grid_result = grid.fit(X, Y.values.ravel())



means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
y_pred_completo = grid.predict(X)

pred = [round(value) for value in y_pred_completo]



y_test_completo = Y.values.ravel()



pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(pred)):

  if pred[x] == y_test_completo[x]:

    if y_test_completo[x] == 1:

      pos_correct += 1

    else:

      neg_correct += 1

       

  if y_test_completo[x] == 1:

    pos_cnt += 1

  else:

    neg_cnt += 1



accuracy = accuracy_score(Y, pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



print("pos_acc", pos_correct/pos_cnt*100, "%", '\tAcertos:', pos_correct, ' de ', pos_cnt)

print("neg_acc", neg_correct/neg_cnt*100, "%", '\tAcertos:', neg_correct, ' de ', neg_cnt)
plot_confusion_matrix(grid, X, Y.values.ravel(), cmap = plt.cm.Blues, values_format = '.10g', display_labels = ['negative', 'positive'])