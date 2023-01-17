# # !pip uninstall sklearn -y

# !pip install -U scikit-learn==0.22.1

# import sklearn

# sklearn.__version__
import numpy as np

import pandas as pd

import scipy

import os, gc

from collections import Counter

from sklearn.model_selection import KFold,StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression

import category_encoders as ce



import warnings

warnings.filterwarnings('ignore')

pd.options.display.max_columns = 50

BIN_COL  = [f'bin_{i}' for i in range(5)]

NOM_COL  = [f'nom_{i}' for i in range(10)]

ORD_COL  = [f'ord_{i}' for i in range(6)]

NOM_5_9  = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

NOM_0_4  = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

DATE_COL = ['day','month']

# from imblearn.over_sampling import RandomOverSampler,SMOTE

import matplotlib.pyplot as plt

import seaborn as sns
%%time



def read_csv():

    train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')

    test  = pd.read_csv('../input/cat-in-the-dat-ii/test.csv')



    train_id = train['id']

    test_id  = test['id']

    train.drop('id', axis=1, inplace=True)

    test.drop('id',  axis=1, inplace=True)

    return train, test, train_id, test_id



def preprocessing(df):

    df.bin_3.replace({'F':0, 'T':1}, inplace=True)

    df.bin_4.replace({'N':0, 'Y':1}, inplace=True)

   

    ord_1_map = {'Novice':1,'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}

    ord_2_map = {'Freezing':1, 'Cold':2,'Warm':3,'Hot':4, 'Boiling Hot':5,'Lava Hot':6}

    df.loc[df['ord_1'].notnull(),'ord_1'] = df.loc[df['ord_1'].notnull(),'ord_1'].map(ord_1_map)

    df.loc[df['ord_2'].notnull(),'ord_2'] = df.loc[df['ord_2'].notnull(),'ord_2'].map(ord_2_map)

    df.loc[df['ord_3'].notnull(),'ord_3'] = df.loc[df['ord_3'].notnull(),'ord_3'].apply(

        lambda c: ord(c) - ord('a') + 1)

    df.loc[df['ord_4'].notnull(),'ord_4'] = df.loc[df['ord_4'].notnull(),'ord_4'].apply(

        lambda c: ord(c) - ord('A') + 1)

    for col in ['ord_1','ord_2','ord_3','ord_4',]:

        df[col] = df[col].astype(np.float32)

    

    df.day = df.day.replace({3:5,2:6,1:7})

    df.loc[df.ord_5.notnull(), 'ord_5_1'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[0])

    df.loc[df.ord_5.notnull(), 'ord_5_2'] = df.loc[df.ord_5.notnull(), 'ord_5'].apply(lambda x: x[1])

    df.loc[df['ord_5_1'].notnull(),'ord_5_1'] = df.loc[df['ord_5_1'].notnull(),'ord_5_1'].apply(

        lambda c: ord(c) - ord('a') + 33).astype(np.float32)

    df.loc[df['ord_5_2'].notnull(),'ord_5_2'] = df.loc[df['ord_5_2'].notnull(),'ord_5_2'].apply(

        lambda c: ord(c) - ord('a') + 33)#.astype(float)

    return df    



def filling_NaN(df):

    df.fillna(-1, inplace=True)

    df.day   = df.day.astype(int)

    df.month = df.month.astype(int)

#     print(df.isnull().sum())

    return df



def target_encoding(cols, smoothing=1.0, min_samples_leaf=1):

    for col in cols:

        encoder = ce.TargetEncoder(cols=col, 

                                   smoothing=smoothing, 

                                   min_samples_leaf=min_samples_leaf)#ce.leave_one_out.LeaveOneOutEncoder()

        train[f'{col}_mean'] = encoder.fit_transform(train[col], train['target'])[col].astype(np.float32)

        test[f'{col}_mean']  = encoder.transform(test[col])[col].astype(np.float32)  

    del encoder

    gc.collect() 
%%time



train, test, train_id, test_id = read_csv()

train = preprocessing(train)

test  = preprocessing(test)

print(f'train day unique value:{train.day.unique()}')

print(f'test  day unique value:{test.day.unique()}')



for col in test.columns:

    if len(set(train[col].dropna().unique().tolist())^ set(test[col].dropna().unique().tolist()))>0:

        train_only = list(set(train[col].dropna().unique().tolist()) - set(test[col].dropna().unique().tolist()))

        test_only  = list(set(test[col].dropna().unique().tolist()) - set(train[col].dropna().unique().tolist()))

        print(col, '(train only)', train_only, '(test only)', test_only) 

        train.loc[train[col].isin(train_only), col] = np.NaN

        test.loc[test[col].isin(test_only), col]    = np.NaN  



target_encoding(['ord_5'])

target_encoding(['ord_5_1'], min_samples_leaf=100)#,'ord_5_2'

# drop_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4','ord_5','ord_5_1']#['ord_5']
%%time



from sklearn.preprocessing import StandardScaler,RobustScaler,MaxAbsScaler,MinMaxScaler





USE_DUMMIES = BIN_COL+NOM_COL+DATE_COL+['ord_0','ord_1', 'ord_2','ord_3', 'ord_4']#, 'ord_5_2'

NOT_DUMMIES = [s for s in train.columns if '_mean' in s]



for col in NOT_DUMMIES:

    scaler = MaxAbsScaler()#MinMaxScaler()#StandardScaler()

    train[col] = scaler.fit_transform(train[[col]]).flatten().astype(np.float32)

    test[col] = scaler.transform(test[[col]]).flatten().astype(np.float32)

del scaler;gc.collect()
%%time

print('drop features:',test.columns.drop(USE_DUMMIES+NOT_DUMMIES).tolist())

target = train['target']

train  = train[USE_DUMMIES+NOT_DUMMIES]   

test   = test[USE_DUMMIES+NOT_DUMMIES]



traintest = pd.concat([train, test], sort=False)

traintest = traintest.reset_index(drop=True)

traintest.head(10)
%%time



traintest = pd.get_dummies(traintest, 

                           columns=USE_DUMMIES,#traintest.columns,

                           dummy_na=False,#True,

                           drop_first=False,#True,

                           sparse=True, 

                           dtype=np.int8 )

traintest.head(10)
%%time



def convert_sparse(traintest, train_length):

#     train_ohe = traintest.iloc[:train_length, :]

#     test_ohe  = traintest.iloc[train_length:, :]

    

#     train_ohe = train_ohe.sparse.to_coo().tocsr().astype(np.float32)

#     test_ohe  = test_ohe.sparse.to_coo().tocsr().astype(np.float32)



    train_ohe = traintest.iloc[:train_length, :]

    test_ohe  = traintest.iloc[train_length:, :]

    

    train_ohe1 = scipy.sparse.bsr_matrix(train_ohe[NOT_DUMMIES])

    test_ohe1  = scipy.sparse.bsr_matrix(test_ohe[NOT_DUMMIES])

    

    train_ohe2 = scipy.sparse.csr_matrix(train_ohe.drop(columns=NOT_DUMMIES))

    test_ohe2  = scipy.sparse.csr_matrix(test_ohe.drop(columns=NOT_DUMMIES))

    

    train_ohe = scipy.sparse.hstack([train_ohe1,train_ohe2]).tocsr()

    test_ohe  = scipy.sparse.hstack([test_ohe1, test_ohe2]).tocsr()

    

    return train_ohe, test_ohe 



train_ohe, test_ohe = convert_sparse(traintest, train_length=len(train))

train_ohe.shape, test_ohe.shape
del train, test;gc.collect()
train_ohe[0:20].todense()
%%time



def print_cv_scores(label, cv_scores):

    print(f'{label} cv scores : {cv_scores}')

    print(f'{label} cv mean score : {np.mean(cv_scores)}')

    print(f'{label} cv std score : {np.std(cv_scores)}')    



def run_cv_model(train_ohe, test_ohe, target, model_fn, params={}, 

                 eval_fn=None, label='model', cv=5,  repeats=5):

    if repeats==1:

        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

#         kf = KFold(n_splits=cv)

        divide_counts = cv

    else:

#         kf = RepeatedKFold(n_splits=cv,n_repeats=repeats, random_state=42)

        kf = RepeatedStratifiedKFold(n_splits=cv,n_repeats=repeats, random_state=42)

        divide_counts = kf.get_n_splits()

        

    fold_splits = kf.split(train_ohe, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train_ohe.shape[0]))

    

    for fold_id, (train_idx, val_idx) in enumerate(fold_splits):

        print(f'Started {label} fold:{fold_id} / {divide_counts}')

        tr_X_ohe, val_X_ohe = train_ohe[train_idx], train_ohe[val_idx]

#         tr_X_ohe, val_X_ohe = train_ohe.iloc[train_idx], train_ohe.iloc[val_idx]

        tr_y, val_y = target[train_idx], target[val_idx]

        print(Counter(tr_y), Counter(val_y))       

        

        params2 = params.copy() 

        model, pred_val_y, pred_test_y = model_fn(

            tr_X_ohe, tr_y, val_X_ohe, val_y, test_ohe, params2)

        

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_idx] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(f'{label} cv score {fold_id}: {cv_score}')

            

    

    print_cv_scores(label, cv_scores)    

    pred_full_test = pred_full_test / divide_counts

    results = {'label': label, 

               'train': pred_train, 

               'test': pred_full_test, 

               'cv': cv_scores}

    return results





def runLR(train_X, train_y, val_X, val_y, test_X, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict val data')

    pred_val_y = model.predict_proba(val_X)[:, 1]

    print('Predict test data')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    return model, pred_val_y, pred_test_y



lr_params = {'penalty':'l2', 

             'solver': 'lbfgs', 'C': 0.05,

#              'class_weight':'balanced', 

             'max_iter':500,#200,#5000

             'random_state':42,

            }



results1 = run_cv_model(

    train_ohe, test_ohe, target, runLR, 

    lr_params, auc, 'lr', cv=10, repeats=2)#5)
%%time



# Make submission

submission = pd.DataFrame(

    {'id': test_id, 'target': results1['test'],})

submission.to_csv('submission.csv', index=False)
plt.figure(figsize=(12,6))

plt.title('distribution of prediction')

sns.distplot(results1['train'])

sns.distplot(results1['test'])

plt.legend(['train','test'])
pd.Series(results1['test']).describe()
submission[:50]
submission[-50:]