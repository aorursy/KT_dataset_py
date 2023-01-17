# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)



train['ord_5_a'] = train['ord_5'].str[0]

train['ord_5_b'] = train['ord_5'].str[1]

test['ord_5_a'] = test['ord_5'].str[0]

test['ord_5_b'] = test['ord_5'].str[1]



train.drop(['ord_5'],inplace=True,axis=1)

test.drop(['ord_5'],inplace=True,axis=1)

test.head()
# Importing categorical options of pandas

from pandas.api.types import CategoricalDtype 

# seting the orders of our ordinal features

ord_3 = CategoricalDtype(categories=['a', 'b', 'c', 'd', 'e', 'f', 'g',

                                     'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'], ordered=True)

ord_4 = CategoricalDtype(categories=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',

                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',

                                     'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], ordered=True)
# Transforming ordinal Features

train.ord_3 = train.ord_3.astype(ord_3).cat.codes

train.ord_4 = train.ord_4.astype(ord_4).cat.codes



# test dataset

test.ord_3 = test.ord_3.astype(ord_3).cat.codes

test.ord_4 = test.ord_4.astype(ord_4).cat.codes



train.head()
def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)



#train.drop(['month','day'],inplace=True,axis=1)

#test.drop(['month','day'],inplace=True,axis=1)

# NOTE, I discovered it on: kaggle.com/gogo827jz/catboost-baseline-with-feature-importance
%%time

# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id','bin_0'], axis=1, inplace=True)

test.drop(['id','bin_0'], axis=1, inplace=True)



print(train.shape)

print(test.shape)
train.head()
one_hot_cols = ['bin_3','bin_4','nom_0', 'nom_1', 'nom_2','ord_0', 'ord_1','ord_2',

       'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_5_a', 'ord_5_b','day','month']

traintest = pd.concat([train, test])



cols_to_scale = ['day_sin','day_cos','month_sin','month_cos','ord_3','ord_4'] 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

traintest[cols_to_scale] = scaler.fit_transform(traintest[cols_to_scale])



dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)

train_ohe = dummies.iloc[:train.shape[0], :]

test_ohe = dummies.iloc[train.shape[0]:, :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time

train_ohe = train_ohe.sparse.to_coo().tocsr()

test_ohe = test_ohe.sparse.to_coo().tocsr()
'''print(train_df_1['ord_1'].unique())

new_dict = {'Grandmaster':5 ,'Expert':4 ,'Novice':1 ,'Contributor':2 ,'Master':3} 

train_df_1['ord_1'] = train_df_1['ord_1'].apply(lambda x: new_dict[x])

test_df_1['ord_1'] = test_df_1['ord_1'].apply(lambda x: new_dict[x])

train_df_1['ord_1'].head()'''
'''train_df_1['ord_3'] = train_df_1[['ord_3', 'ord_4']].apply(lambda x: ''.join(x), axis=1)

train_df_1.drop(['ord_4'], axis=1, inplace=True)

train_df_1.loc[:,'ord_1':'target'].head()



test_df_1['ord_3'] = test_df_1[['ord_3', 'ord_4']].apply(lambda x: ''.join(x), axis=1)

test_df_1.drop(['ord_4'], axis=1, inplace=True)



val_ord3 = train_df_1['ord_3'].value_counts()

val_ord5 = train_df_1['ord_5'].value_counts()

train_df_1['ord_3'] = train_df_1.loc[:,'ord_3'].apply(lambda x: val_ord3[x])

train_df_1['ord_5'] = train_df_1.loc[:,'ord_5'].apply(lambda x: val_ord5[x])

print(train_df_1.loc[:,'ord_1':].head())



val_ord3 = test_df_1['ord_3'].value_counts()

val_ord5 = test_df_1['ord_5'].value_counts()

test_df_1['ord_3'] = test_df_1.loc[:,'ord_3'].apply(lambda x: val_ord3[x])

test_df_1['ord_5'] = test_df_1.loc[:,'ord_5'].apply(lambda x: val_ord5[x])'''
%%time



from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier



# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=5)#,shuffle=True,random_state=0)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    #print('\n\n\n',pred_full_test,'\n\n')

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2



def runSGD(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = SGDClassifier(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return pred_test_y, pred_test_y2



lr_params = {'solver': 'lbfgs', 'C': 0.096, 'max_iter':500, 'class_weight':'balanced'}

sgd_params = {'loss': 'log', 'penalty': 'l2', 'max_iter':1000}

results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')
submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission_11.csv', index=False)
'''from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500,verbose=True,activation='tanh',solver='sgd',alpha=0.003,learning_rate='adaptive')

mlp.fit(xTrain,yTrain)



from sklearn.metrics import roc_auc_score

prediction = mlp.predict(test_df_1[:])

roc_auc_score(yTest, predict_yTest)

print(prediction.sum())'''