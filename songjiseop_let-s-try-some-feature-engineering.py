%%time



import pandas as pd

import numpy as np



# Load data

train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')



print(train.shape)

print(test.shape)
%%time



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['target', 'id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



print(train.shape)

print(test.shape)
def count_transform(column):

    return column.map(column.value_counts().to_dict())
rang = {"Grandmaster" : 4, "Master" : 3, "Expert" : 2, "Contributor" : 1, "Novice" : 0}

temperature = {"Freezing" : 0, "Cold": 1, "Warm" : 2, "Hot": 3, "Boiling Hot" : 4, "Lava Hot" : 5}
traintest = pd.concat([train, test])
%%time

from scipy.sparse import csr_matrix, hstack

# One Hot Encode

traintest['ord_1_new'] = traintest['ord_1'].map(rang)

traintest['ord_2_new'] = traintest['ord_2'].map(temperature)

traintest['ord_3_new'] = traintest['ord_3'].map({val : idx for idx, val in enumerate(np.unique(traintest['ord_3']))})

traintest['ord_4_new'] = traintest['ord_4'].map({val : idx for idx, val in enumerate(np.unique(traintest['ord_4']))})

traintest['ord_5_new_1'] = traintest['ord_5'].apply(lambda x: x[0])

traintest['ord_5_new_1'] = traintest['ord_5_new_1'].map({val : idx for idx, val in enumerate(np.unique(traintest['ord_5_new_1']))})

traintest['ord_5_new_2'] = traintest['ord_5'].apply(lambda x: x[1])

traintest['ord_5_new_2'] = traintest['ord_5_new_2'].map({val : idx for idx, val in enumerate(np.unique(traintest['ord_5_new_2']))})

traintest['ord_5_new_2'] = traintest['ord_5_new_2'].map({val : idx for idx, val in enumerate(np.unique(traintest['ord_5_new_2']))})

#traintest['new_month_sin'] = np.sin(2 * np.pi * traintest['month']/12.0)

#traintest['new_month_sin'] = np.cos(2 * np.pi * traintest['month']/12.0)

#traintest['new_day_sin'] = np.sin(2 * np.pi * traintest['day']/7.0)

#traintest['new_day_sin'] = np.sin(2 * np.pi * traintest['day']/7.0)



dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True).to_sparse().to_coo()

#count_encode = csr_matrix(traintest.apply(count_transform))

traintest_new = csr_matrix(traintest[list(filter(lambda x: 'new' in x, traintest.columns))])
feature_df = hstack([dummies, traintest_new]).tocsr()
train_idx = np.array(list(train.index))

train_ohe = feature_df[train_idx, :]

test_ohe = feature_df[np.max(train_idx) + np.array(list(test.index)), :]



print(train_ohe.shape)

print(test_ohe.shape)
%%time

from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression, Lasso, Ridge

from sklearn.naive_bayes import BernoulliNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

# Model

def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    coefs = []

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train[dev_index], train[val_index]

        dev_y, val_y = target[dev_index], target[val_index]

        params2 = params.copy()

        trn_res = model_fn(dev_X, dev_y, val_X, val_y, test, params2)

        pred_val_y, pred_test_y = trn_res['pred_val_y'], trn_res['pred_test_y']

        pred_full_test = pred_full_test + pred_test_y

        pred_train[val_index] = pred_val_y

        if eval_fn is not None:

            cv_score = eval_fn(val_y, pred_val_y)

            cv_scores.append(cv_score)

            coefs.append(trn_res['coef'])

            print(label + ' cv score {}: {}'.format(i, cv_score))

        i += 1

    print('{} cv scores : {}'.format(label, cv_scores))

    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))

    print('{} cv std score : {}'.format(label, np.std(cv_scores)))

    pred_full_test = pred_full_test / 5.0

    results = {'label': label,

              'train': pred_train, 'test': pred_full_test,

              'cv': cv_scores, 

              'coefs' : coefs}

    return results





def runLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = LogisticRegression(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict_proba(test_X)[:, 1]

    print('Predict 2/2')

    pred_test_y2 = model.predict_proba(test_X2)[:, 1]

    return {'pred_val_y' : pred_test_y, 'pred_test_y' : pred_test_y2, 'coef' : model.coef_}



def runRLR(train_X, train_y, test_X, test_y, test_X2, params):

    print('Train LR')

    model = Ridge(**params)

    model.fit(train_X, train_y)

    print('Predict 1/2')

    pred_test_y = model.predict(test_X)

    print('Predict 2/2')

    pred_test_y2 = model.predict(test_X2)

    return pred_test_y, pred_test_y2

rr_params = {'alpha' : 1, 'solver': 'lsqr', "fit_intercept" : False}

#rr_params = {'alpha' : 1, 'solver': 'sparse_cg'}

lr_params = {'solver': 'lbfgs', 'C': 0.1,'max_iter' : 1000}

results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'Ridge')
%%time



# We now have a model with a CV score of 0.8032. Nice! Let's submit that



# Make submission

submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission.csv', index=False)