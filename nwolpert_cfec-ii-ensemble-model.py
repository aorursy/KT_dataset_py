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
import subprocess

import sys

# for uninstalled packages, use:

def install(package):

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
import gc



from matplotlib import pyplot as plt

import seaborn as sns



import scipy



from scipy.sparse import csr_matrix

from scipy.sparse import hstack

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

import category_encoders as ce



from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Ridge

from catboost import CatBoostClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV

import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.metrics import auc, roc_curve, roc_auc_score
## Memory optimization



# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin

# Modified by @Vopani



# to support timestamp type, categorical type and to add option to use float16

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype



def reduce_mem_usage(df, use_float16=False):

    """

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        

    """

    

    start_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == "int":

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# function to show memory usage of data

BYTES_TO_MB_DIV = 0.000001

def print_memory_usage_of_data_frame(df):

    mem = round(df.memory_usage().sum() * BYTES_TO_MB_DIV, 3) 

    print("Memory usage is " + str(mem) + " MB")
path_data = '/kaggle/input/cat-in-the-dat-ii/'

train = pd.read_csv(path_data + 'train.csv')

test = pd.read_csv(path_data + 'test.csv')

#sample_submission = pd.read_csv(path_data + 'sample_submission.csv')

size_train = train.shape[0]

size_test = test.shape[0]
train = reduce_mem_usage(train, use_float16=True)

test = reduce_mem_usage(test, use_float16=True)
# merge the data

target = train.target

train_id = train['id']

test_id  = test['id']

train_test = pd.concat([train, test], sort=False).drop(columns=['target', 'id'])

train_test.shape
binary_columns = [col for col in train_test if col.startswith('bin')]

nominal_columns = [col for col in train_test if col.startswith('nom')]

ordinal_columns = [col for col in train_test if col.startswith('ord')]
train.head(10)
pd.set_option('display.max_columns', 500)

train.head(10)
test.head(10)
print('Shape of training data: ', train.shape)

print('Shape of testing data: ', test.shape)
train.describe()
train.dtypes
total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)

missing_train  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

total = test.isnull().sum().sort_values(ascending = False)

percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)

missing_test  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train
fig = plt.figure(figsize=(20,10))

fig1 = fig.add_subplot(221)

missing_train['Total'].plot.bar(x='lab', y='val', rot=45)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

fig1.set_title('Total missing values train', fontsize=20)



fig2 = fig.add_subplot(222)

missing_train['Percent'].plot.bar(x='lab', y='val', rot=45)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.ylabel('percentage', fontsize=12)

fig2.set_title('Percentage missing values train', fontsize=20)



fig3 = fig.add_subplot(223)

missing_train['Total'].plot.bar(x='lab', y='val', rot=45)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

fig3.set_title('Total missing values test', fontsize=20)



fig4 = fig.add_subplot(224)

missing_train['Percent'].plot.bar(x='lab', y='val', rot=45)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.ylabel('percentage', fontsize=12)

fig4.set_title('Percentage missing values test', fontsize=20)



plt.tight_layout()
# target distribution

train['target'].value_counts().plot(kind='bar', title='target value distribution')
# Show which features are correlated among each other

# create helper function to show a correlation plot between the different features

def plot_correlation_heatmap(df):

    

    corr = df.corr()



    sns.set(style="white")

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    f, ax = plt.subplots(figsize=(11, 9))

    #cmap = sns.diverging_palette(220, 10, as_cmap=True)

    cmap = 'coolwarm'



    sns.heatmap(corr, mask=mask, cmap=cmap,  center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()

    

    return corr



corr = plot_correlation_heatmap(train)
f, axes = plt.subplots(2, 3, figsize=(20, 10))

for ax, i in zip(axes.flatten(), range(5)):

    sns.countplot(x='bin_' + str(i), hue='target', data= train, ax=ax)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.tight_layout()
f, axes = plt.subplots(2, 3, figsize=(20, 10))

for ax, i in zip(axes.flatten(), range(5)):

    sns.countplot(x='nom_' + str(i), hue='target', data= train, ax=ax)

plt.tight_layout()
f, axes = plt.subplots(2, 3, figsize=(20, 10))

for ax, i in zip(axes.flatten(), range(5)):

    sns.countplot(x='ord_' + str(i), hue='target', data= train, ax=ax)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.tight_layout()
f, axes = plt.subplots(2, 1, figsize=(20, 10))



sns.countplot(x='day', hue='target', data= train, ax=axes[0])

axes[0].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.tight_layout()



sns.countplot(x='month', hue='target', data= train, ax=axes[1])

axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.tight_layout()
# show number of unique values per column

print('###### Nominal columns: #######')

for col in nominal_columns:

    print('Number of values for column ' + col + ': ' + str(train_test[col].nunique()))

print('###### Ordinal columns: #######')

for col in ordinal_columns:

    print('Number of values for column ' + col + ': ' + str(train_test[col].nunique()))
print_memory_usage_of_data_frame(train)
# replace nans by most frequent values

def replace_nan(data):

    for column in data.columns:

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])





replace_nan(train_test)

train_test.head(10)
train_test.bin_3.replace({'F':0, 'T':1}, inplace=True)

train_test.bin_4.replace({'N':0, 'Y':1}, inplace=True)
map_ord_1 = {'Novice':1, 'Contributor':2, 'Expert':4, 'Master':5, 'Grandmaster':6}

train_test.ord_1 = train_test.ord_1.map(map_ord_1)



map_ord_2 = {'Freezing':1, 'Cold':2, 'Warm': 3, 'Hot':4, 'Boiling Hot':5, 'Lava Hot':6}

train_test.ord_2 = train_test.ord_2.map(map_ord_2)



map_ord_3 = {key:value for value,key in enumerate(sorted(train_test.ord_3.dropna().unique()))} 

train_test.ord_3 = train_test.ord_3.map(map_ord_3)



map_ord_4 = {key:value for value,key in enumerate(sorted(train_test.ord_4.dropna().unique()))} 

train_test.ord_4 = train_test.ord_4.map(map_ord_4)



map_ord_5 = {key:value for value,key in enumerate(sorted(train_test.ord_5.dropna().unique()))} 

train_test.ord_5 = train_test.ord_5.map(map_ord_5)
train_test.head()
dummies_nominal = pd.get_dummies(train_test[nominal_columns], columns=nominal_columns, drop_first=True, sparse=True)



# update nominal columns

nominal_columns = [col for col in train_test if col.startswith('nom')]



# split back into train and test

dummies_nominal_train = dummies_nominal.iloc[:size_train, :]

dummies_nominal_test  = dummies_nominal.iloc[size_train:, :]



dummies_nominal_train.head()
dummies_nominal_test.head()
train_test
# nominal dummy variables

dummies_nominal_train = dummies_nominal_train.sparse.to_coo().tocsr()

dummies_nominal_test = dummies_nominal_test.sparse.to_coo().tocsr()

dummies_nominal_train
# all other variables

train = train_test.iloc[:size_train, :]

test  = train_test.iloc[size_train:, :]

train = csr_matrix(train.drop(nominal_columns, axis=1).astype('float').values)

test = csr_matrix(test.drop(nominal_columns, axis=1).astype('float').values)

train
# append nominal to train_test

train = hstack((train,dummies_nominal_train)).tocsr()

test = hstack((test,dummies_nominal_test)).tocsr()

train
test
def search_params(model, params, label):

    

    print('Searching hyparameters for ' + label + ' model...')

    clf = GridSearchCV(model, scoring='roc_auc', param_grid = params, cv = 5, verbose=True, n_jobs=-1)

    clf.fit(train, target)

    print('Tuned hyperparameters :(best parameters) ',clf.best_params_)

    print('Accuracy :',clf.best_score_)

    

    return clf.best_params_,clf.best_score_
def train_model(model,label):

    

    folds = 10

    kf = KFold(n_splits=folds)

    scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    fig= plt.figure(figsize=(15,20))

    for ifold, (dev_index, val_index) in enumerate(kf.split(train, target)):

        print('Started fold ' + str(ifold+1) + '/10')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        

        model.fit(dev_X, dev_y)

        print('Predict validation')

        if label=='ridge':

            pred_val_y = model.predict(val_X)

            pred_test_y = model.predict(test)

            pred_full_test += pred_test_y/folds

        else:

            pred_val_y = model.predict_proba(val_X)[:, 1]

            pred_test_y = model.predict_proba(test)[:, 1]

            pred_full_test = pred_full_test + pred_test_y

        print('Predict test')

        pred_train[val_index] = pred_val_y

        auc_score = roc_auc_score(val_y, pred_val_y)

        scores.append(auc_score)

        print('cv score {}: {}'.format(ifold+1, auc_score))

        

        plt.subplot(5, 3, ifold+1)

        fpr, tpr, thresholds = roc_curve(val_y, pred_val_y)

        plt.plot(fpr, tpr)

        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0.0, 1.0])

        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate')

        plt.ylabel('True Positive Rate')

        plt.title('Fold=%i' %i + ', AUC=%f' %auc_score)

    plt.tight_layout(pad=2)

    plt.show()

    print('{} cv auc scores : {}'.format(label, scores))

    print('{} cv mean auc score : {}'.format(label, np.mean(scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': scores}

    return results
'''

params_lr = {

    'solver': ['lbfgs'],

    'max_iter':[10000],

    'C' : np.logspace(-4, 4, 20),

    'penalty' : ['l1', 'l2']

}



best_params_lr = search_params(LogisticRegression(), params_lr, 'lr')

'''
'''

params_ridge = {

        'alpha': [1e-3, 1e-2, 1e-1, 1]

}



best_params_ridge = search_params(Ridge(), params_ridge, 'rd')

'''
'''

params_catboost = {

        'max_depth': [2, 3, 4, 5],

        'n_estimators': [50, 100, 200, 400, 600],

        'random_state': [42],

        'verbose': [0]

}



best_params_catboost = search_params(CatBoostClassifier(), params_catboost, 'cb')

'''
'''

params_xgb = {

        'learning_rate': [0.01, 0.05, 0.1],

        'max_depth': [2, 3, 5],

        'n_estimators': [1000],

        'subsample': [0.8],

        'random_state': [42],

        'verbosity': [0],

        'objective': ['binary:logistic']

}



best_params_xgb = search_params(XGBClassifier(), params_xgb, 'xgb')

'''
lr = LogisticRegression(solver='lbfgs', C = 0.08858667904100823, max_iter = 10000, penalty = 'l2')

results_lr = train_model(lr, 'logregress')
rd = Ridge(alpha = 1)

results_rd = train_model(rd, 'ridge')
cb = CatBoostClassifier(max_depth=5, n_estimators=600, random_state=42, verbose=0)

results_cb = train_model(cb, 'catboost')
xgb = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=1000,

                                objective ='binary:logistic',

                                subsample = 0.8, verbosity=0)

results_xgb = train_model(xgb, 'XGBClassifier')
results_lr
# Dataset that will be the train set of the ensemble model.

predictions_first_level_train = pd.DataFrame(results_lr['train'], columns=['logistic_regression'])

predictions_first_level_train['ridge_regression'] = results_rd['train']

predictions_first_level_train['catboost'] = results_cb['train']

predictions_first_level_train['xgboost'] = results_xgb['train']

predictions_first_level_train.head(20)



predictions_first_level_test = pd.DataFrame(results_lr['test'], columns=['logistic_regression'])

predictions_first_level_test['ridge_regression'] = results_rd['test']

predictions_first_level_test['catboost'] = results_cb['test']

predictions_first_level_test['xgboost'] = results_xgb['test']

predictions_first_level_test.head()
# 2nd order model

meta_xgb = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.01, n_jobs=-1,

                                objective ='binary:logistic',subsample = 0.8, verbosity=0)



nfolds = 5

kf = KFold(n_splits=nfolds)

scores = []

pred_final_train = np.zeros((predictions_first_level_train.shape[0]))

pred_final_test = np.zeros((nfolds, predictions_first_level_test.shape[0]))

for ifold, (dev_index, val_index) in enumerate(kf.split(predictions_first_level_train, target)):

    print('Started fold ' + str(ifold+1) + '/10')

    dev_X, val_X = predictions_first_level_train.loc[dev_index, :], predictions_first_level_train.loc[val_index, :]

    dev_y, val_y = target[dev_index], target[val_index]

    

    meta_xgb.fit(dev_X, dev_y)

    print('Predict validation')

    pred_val_y = meta_xgb.predict_proba(val_X)[:, 1]

    pred_final_test[ifold] = meta_xgb.predict_proba(predictions_first_level_test)[:, 1]

    print('Predict test')

    pred_final_train[val_index] = pred_val_y

    auc_score = roc_auc_score(val_y, pred_val_y)

    scores.append(auc_score)

    print('cv score {}: {}'.format(ifold+1, auc_score))

pred_mean_score = roc_auc_score(target, pred_final_train)

print('Final model cv mean auc score : {}'.format(pred_mean_score))

pred_final_test=pred_final_test.mean(axis=0)
pred_final_test
submission = pd.DataFrame({'id': test_id, 'target': pred_final_test})

submission.to_csv('submission.csv', index=False)