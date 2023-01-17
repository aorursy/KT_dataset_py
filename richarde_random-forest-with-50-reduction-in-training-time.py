# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', parse_dates=['project_submitted_datetime'])
test = pd.read_csv('../input/test.csv', parse_dates=['project_submitted_datetime'])
resources = pd.read_csv('../input/resources.csv')
#submission = pd.read_csv('sample_submission.csv')
train.head(1)
resources.head()
mem_df1 = train[['id','teacher_id','teacher_number_of_previously_posted_projects','project_is_approved']]
resources['total_cost'] = resources['quantity']*resources['price']
mem_df2 = resources[['id','quantity','price','total_cost']]
train_mem = pd.merge(mem_df1,mem_df2,how='left',on='id')
train_mem.info()
#some examples of datatypes unsigned and signed
data_types = ["uint8","int8","int16","uint16","uint64","int64"]
for it in data_types:
    print(np.iinfo(it))
train_mem.describe()
def mem_usage(pandas_obj):
    usage_b = pandas_obj.memory_usage(deep=True).sum()
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

train_mem_int = train_mem.select_dtypes(include=['int'])
converted_int = train_mem_int.apply(pd.to_numeric,downcast='unsigned')

print("Size of integer types before {}".format(mem_usage(train_mem_int)))
print("Size of integer types after {}".format(mem_usage(converted_int)))

compare_ints = pd.concat([train_mem_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['No. of types before','No. of types after']
compare_ints.apply(pd.Series.value_counts)
train_mem_float = train_mem.select_dtypes(include=['float'])
converted_float = train_mem_float.apply(pd.to_numeric,downcast='float')

print("Size of float types before: {}".format(mem_usage(train_mem_float)))
print("Size of float types after: {}".format(mem_usage(converted_float)))

compare_floats = pd.concat([train_mem_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['No. of types before','No. of types after']
compare_floats.apply(pd.Series.value_counts)
X = train_mem.drop(['id', 'project_is_approved','teacher_id'], axis=1)
y = train_mem['project_is_approved']
print(X.dtypes)
print(" ")
print(y.dtypes)
num_folds = 5
seed = 7
scoring = 'accuracy'

start = time.time()
param_grid = {'max_depth': [5,8],
              'min_samples_split':[3,5]
             }

model = RandomForestClassifier(n_jobs=-1)
kfold =KFold(n_splits = num_folds,random_state = seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X,y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
time_first = end - start
columns_to_overwrite_float = ['total_cost','price']
train_mem.drop(labels=columns_to_overwrite_float, axis="columns", inplace=True)
train_mem[columns_to_overwrite_float] = converted_float[columns_to_overwrite_float]
columns_to_overwrite_int = ['teacher_number_of_previously_posted_projects','project_is_approved','quantity']
train_mem.drop(labels=columns_to_overwrite_int, axis="columns", inplace=True)
train_mem[columns_to_overwrite_int] = converted_int[columns_to_overwrite_int]
train_mem_after = train_mem.copy(deep=True)
train_mem_after.info()
X = train_mem_after.drop(['id', 'project_is_approved','teacher_id'], axis=1)
y = train_mem_after['project_is_approved']
print(X.dtypes)
print(" ")
print(y.dtypes)
start = time.time()

param_grid = {'max_depth': [5,8],
              'min_samples_split':[3]
             }
model = RandomForestClassifier(n_jobs=-1)
kfold =KFold(n_splits = num_folds,random_state = seed)
grid = GridSearchCV(estimator = model,param_grid = param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X,y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))
time_second = end - start
plt.figure(figsize=(10,5))
x_pos = [0,1]
x_label= ['before','after']
scores = [time_first,time_second]
plt.bar(x_pos,scores,align='center')
plt.xlabel('Memory Optimisation',fontsize=12)
plt.xticks(x_pos,x_label)
plt.ylabel('Time taken in secs',fontsize=12)
plt.show();
