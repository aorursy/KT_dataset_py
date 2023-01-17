import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



PATH = '/kaggle/input/cat-in-the-dat/'
#Reading the dataset.

train_df = pd.read_csv(f'{PATH}train.csv', index_col='id')

test_df = pd.read_csv(f'{PATH}test.csv', index_col='id')
#exploring the train dataset.

train_df.head()
#shape of the datasets.

print(f'Training shape: {train_df.shape}')

print(f'Testing shape: {test_df.shape}')
#first checking the categrical variables containing in the train sets are also present in the test set.

def checkcat(df):

    for col in df.columns:

        length = len(set(test_df[col].values) - set(train_df[col].values))

        if length > 0:

            print(f'{col} in the test set has {length} values that are not present in the train set')

checkcat(test_df)
#One Hot Encoding binary features.

cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



train_df = pd.get_dummies(train_df, columns=cols)

test_df = pd.get_dummies(test_df, columns=cols)
#Label encoding nominal features.

from sklearn.preprocessing import LabelEncoder



#Label encoding everything.



for col in train_df.columns:

    if train_df[col].dtype == 'O':

        #initializing.

        le = LabelEncoder()

        le.fit(list(train_df[col].values) + list(test_df[col].values))

        train_df[col] = le.transform(list(train_df[col].values))

        test_df[col] = le.transform(list(test_df[col].values))
train_df.head()
#encoding the cyclic features.

#refrence : https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning

def encode(data, col, max_val):

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)

    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

    return data



#day

train_df = encode(train_df, 'day', 6)

test_df = encode(test_df, 'day', 6)



#month.

train_df = encode(train_df, 'month', 12)

test_df = encode(test_df, 'month', 12)
#dropping the day month columns.

train_df.drop(['day', 'month'], axis=1, inplace=True)

test_df.drop(['day', 'month'], axis=1, inplace=True)
#Creating X and y variables.

X = train_df.drop('target', axis=1)

y = train_df.target
# #scaling the dataset.

# from sklearn.preprocessing import StandardScaler



# #initializing.

# scale = StandardScaler()



# #fit

# X = scale.fit_transform(X)

# test_df = scale.transform(test_df)
print('#'*20)

print('StratifiedKFold training...')



# Same as normal kfold but we can be sure

# that our target is perfectly distribuited

# over folds

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

import catboost as cg

from sklearn.metrics import roc_auc_score



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)



#initializing the model

model = cg.CatBoostClassifier(logging_level='Silent')



#score.

score = []



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=y)):

    print('Fold:',fold_+1)

    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    

    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]

    

    #fitting on training data.

    %time model.fit(tr_x, tr_y)

    

    #predicting on test.

    y_pred = model.predict(vl_x)

    

    #storing score

    score.append(roc_auc_score(v_y, y_pred))

    print(f'AUC score : {roc_auc_score(v_y, y_pred)}')



print('Average AUC score', np.mean(score))

print('#'*20)
#fitting on the entire data.

%time model.fit(X, y)
#making predictions on test data.

pred_test = model.predict_proba(test_df)[:,0]
#submission file.

sub = pd.read_csv(f'{PATH}sample_submission.csv')

# #reseting index

# test_df = test_df.reset_index()

sub['target'] = pred_test

sub.to_csv('catboost_model_0.1.csv', index=None, header=True)