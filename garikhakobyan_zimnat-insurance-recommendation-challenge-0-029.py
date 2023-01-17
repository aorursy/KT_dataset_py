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
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import copy
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


from itertools import  combinations
train = pd.read_csv('/kaggle/input/zimnat-insurance-recommendation-challenge/Train.csv', parse_dates=['join_date'])
test = pd.read_csv('/kaggle/input/zimnat-insurance-recommendation-challenge/Test.csv', parse_dates=['join_date'])
submission = pd.read_csv('/kaggle/input/zimnat-insurance-recommendation-challenge/SampleSubmission.csv')
train['Product_QTY'] = np.sum(train.iloc[:,8:], axis = 1)
test['Product_QTY'] = np.sum(test.iloc[:,8:], axis = 1)

test['Product_QTY'] = test['Product_QTY'].apply(lambda x: x+1)


train['year'] = train['join_date'].apply(lambda x: x.year)
test['year'] = test['join_date'].apply(lambda x: x.year)

train.drop(['join_date'], axis = 1, inplace = True)
test.drop(['join_date'], axis = 1, inplace = True)

train.fillna(method = 'bfill', inplace = True)
test.fillna(method = 'bfill', inplace = True)
cols = ['ID', 'sex', 'marital_status', 'birth_year', 'branch_code', 'occupation_code', 
        'occupation_category_code','Product_QTY', 'year', 'P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR',
        'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D',
        'J9JW', 'GHYX', 'ECY3' ]

train = train[cols].copy()
test = test[cols].copy()
X_train = []
all_features = []
X_train_columns = train.columns
c = 0
for v in train.values:
    info = v[:9]
    binary = v[9:]
    index = [k for k, i in enumerate(binary) if i == 1]
    all_features.append(index)
    for i in index:
#         all_features.append(all_features)
        c+=1
        for k in range(len(binary)):
            if k == i:
                binary_transformed = list(copy.copy(binary))
                binary_transformed[i] = 0
                X_train.append(list(info) + binary_transformed + [X_train_columns[9+k]])

X_train = pd.DataFrame(X_train)
X_train.columns = ['ID', 'sex', 'marital_status', 'birth_year', 'branch_code', 'occupation_code', 
        'occupation_category_code','Product_QTY', 'year', 'P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR',
        'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D',
        'J9JW', 'GHYX', 'ECY3', 'label' ]
y_train = X_train.pop('label')
X_test = []
true_values = []
c = 0
for v in test.values:
    c += 1
    info = v[:9]
    binary = v[9:]
    index = [k for k, i in enumerate(binary) if i == 1]
    X_test.append(list(info) + list(binary))
    for k in test.columns[9:][index]:
        true_values.append(v[0] + ' X ' + k)

X_test = pd.DataFrame(X_test)
X_test.columns = ['ID', 'sex', 'marital_status', 'birth_year', 'branch_code', 'occupation_code', 
        'occupation_category_code','Product_QTY', 'year', 'P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR',
        'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 'N2MW', 'AHXO', 'BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D',
        'J9JW', 'GHYX', 'ECY3']
features_train = []
features_test = []
columns = []

append_features = ['ID','P5DA', 'RIBP', '8NN1', '7POT', '66FJ', 'GYSR', 'SOP4', 'RVSZ', 'PYUQ', 'LJR9', 
'N2MW', 'AHXO','BSTQ', 'FM3X', 'K6QO', 'QBOL', 'JWFN', 'JZ9D', 'J9JW', 'GHYX', 
'ECY3',  'Product_QTY', 'year','sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code',
'birth_year']
for v in append_features:
    features_train.append(X_train[v].values.reshape(-1, 1))
    features_test.append(X_test[v].values.reshape(-1, 1))
    columns.append(np.array([v]))


features_train = np.concatenate(features_train, axis=1)
features_test = np.concatenate(features_test, axis=1)
columns = np.concatenate(np.array(columns))

X_train = pd.DataFrame(features_train)
X_train.columns = columns
X_test = pd.DataFrame(features_test)
X_test.columns = columns

le = LabelEncoder()
data = X_train.append(X_test)
for v in X_train.select_dtypes('object').columns[1:]:
    data.loc[:,v] = le.fit_transform(data.loc[:,v])
X_train = data[:X_train.shape[0]]
X_test = data[-X_test.shape[0]:]
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
y_train = y_train.fillna(0)
X_test
ytrain = LabelEncoder()
y_train = ytrain.fit_transform(y_train)
cat_features = ['sex', 'marital_status', 'birth_year', 'branch_code', 'occupation_code', 'occupation_category_code']
1800
1500
1200
1000
800
models = []
models.append(CatBoostClassifier(random_state=17, num_trees=1800, max_depth = 3, cat_features = cat_features, task_type= 'GPU'))
models.append(CatBoostClassifier(random_state=18, num_trees=1500, max_depth=5, cat_features = cat_features, task_type= 'GPU'))
models.append(CatBoostClassifier(random_state=19, num_trees=1200, max_depth = 6, cat_features = cat_features, task_type= 'GPU'))
models.append(CatBoostClassifier(random_state=20, num_trees=1000, max_depth=7, cat_features = cat_features, task_type= 'GPU'))
models.append(CatBoostClassifier(random_state=21, num_trees=800, max_depth=8, cat_features = cat_features, task_type= 'GPU'))

kf = KFold(n_splits=5, random_state = 101,  shuffle = True)

result =pd.DataFrame()
submission_cluster = pd.DataFrame()

new = 0
for fold, (train_idx, test_idx) in tqdm(enumerate(kf.split(X_train))):
    train = X_train.loc[X_train.index[train_idx]].copy()
    test =  y_train[train_idx].copy()
    

    for counter in tqdm(range(len(models))):
        models[counter].fit(train.drop('ID', axis = 1), test, verbose = False)

        proba = models[counter].predict_proba(X_test.drop('ID', axis = 1))
        y_test = pd.DataFrame(proba)
        y_test.columns = ytrain.inverse_transform(y_test.columns)


        answer_mass = []
        for i in range(X_test.shape[0]):
            id = X_test['ID'].iloc[i]
            for c in y_test.columns:
                answer_mass.append([id + ' X ' + c, y_test[c].iloc[i]])


        df_answer = pd.DataFrame(answer_mass)
        df_answer.columns = ['ID X PCODE', 'Label']
        
#         print('goes to submission_cluster--',df_answer.shape)
        

        if new == 0:
            submission_cluster = pd.concat([submission_cluster, df_answer])
        else:
            submission_cluster = pd.merge(submission_cluster, df_answer, on = 'ID X PCODE')


        print('Fold--', fold, ' Counter--', counter, ' Score--', models[counter].best_score_ )    
   
#     result = pd.concat([result,submission_cluster])     
    
        new += 1
print('Sape of output--', submission_cluster.shape)        
submission_cluster['Label'] = submission_cluster.iloc[:,1:].mean(axis = 1)

# submission_cluster['Label'].sum()
# submission_cluster['Label'] = submission_cluster['Label_'].apply(lambda x: 0 if x < 0.025 else x)
submission_cluster = submission_cluster[['ID X PCODE', 'Label']].copy()
for i in tqdm(range(submission_cluster.shape[0])):
    if submission_cluster['ID X PCODE'].iloc[i] in true_values:
        submission_cluster['Label'].iloc[i] = 1.0

submission_cluster[['ID X PCODE', 'Label']].to_csv('SampleDabmission.csv', index = False)