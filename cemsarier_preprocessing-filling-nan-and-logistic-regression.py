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
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")



# Subset

target = train['target']

train_id = train['id']

test_id = test['id']

train.drop(['id'], axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



#Null values

null_df = pd.DataFrame({'Percentile':train.isnull().sum()/len(train), 'Count':train.isnull().sum()})

print(null_df)
my_df = pd.concat([train, test])

my_df.isnull().sum()
my_df["bin_3"] = my_df["bin_3"].apply(lambda x: 1 if x=='T' else (0 if x=='F' else None))

my_df["bin_4"] = my_df["bin_4"].apply(lambda x: 1 if x=='Y' else (0 if x=='N' else None))
for enc in ["nom_0","nom_1","nom_2","nom_3","nom_4","day","month","ord_3"]:#,"ord_4","ord_5"]:

    my_df[enc] = my_df[enc].astype("str")

    my_df[enc] = my_df[enc].apply(lambda x: x.lower())

    my_df[enc] = my_df[enc].apply(lambda x: None if x=='nan' else x)
for enc in ["ord_4","ord_5"]:

    my_df[enc] = my_df[enc].astype("str")

    my_df[enc] = my_df[enc].apply(lambda x: None if x=='nan' else x)
my_df["ord_1"] = my_df["ord_1"].apply(lambda x: 1 if x=='Novice' else (2 if x=='Contributor' else (3 if x=='Expert' else (4 if x=='Master' else (5 if x=='Grandmaster' else None)))))

my_df["ord_2"] = my_df["ord_2"].apply(lambda x: 1 if x=='Freezing' else (2 if x=='Cold' else (3 if x=='Warm' else (4 if x=='Hot' else (5 if x=='Boiling Hot' else (6 if x=='Lava Hot' else None))))))

for col in ["nom_5","nom_6","nom_7","nom_8","nom_9"]:

    mode = my_df[col].mode()[0]

    my_df[col] = my_df[col].astype(str)

    my_df[col] = my_df[col].apply(lambda x: mode if x=='nan' else x)

    
columns = list(my_df.columns)

for col in ["target","nom_5","nom_6","nom_7","nom_8","nom_9"]:

    columns.remove(col)
for col in columns:

    my_df[col] = my_df.groupby(["nom_7"])[col].transform(lambda x: x.fillna(x.mode()[0]))
my_df.isnull().sum()
from sklearn.preprocessing import OrdinalEncoder

oencoder = OrdinalEncoder(dtype=np.int16)

for enc in ["ord_3","ord_4","ord_5"]:

    my_df[enc] = oencoder.fit_transform(np.array(my_df[enc]).reshape(-1,1))
for category in ["nom_5","nom_6","nom_7","nom_8","nom_9"]:

    print("{} has {} unique values".format(category,len(np.unique(my_df[category]))))
for enc in ["nom_0","nom_1","nom_2","nom_3","nom_4","day","month","nom_7","nom_8"]:

    enc1 = pd.get_dummies(my_df[enc], prefix=enc)

    my_df.drop(columns=enc, inplace=True)

    my_df = pd.concat([my_df,enc1], axis=1)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler((-1,1))

for feat in ["ord_0","ord_1","ord_2","ord_3","ord_4","ord_5"]:

    my_df[feat] = scaler.fit_transform(np.array(my_df[feat]).reshape(-1,1))





for feat in ["bin_0","bin_1","bin_2","bin_3","bin_4"]:

    my_df[feat] = scaler.fit_transform(np.array(my_df[feat]).reshape(-1,1))
test = my_df[my_df["target"].isnull()]

test.drop(columns='target', inplace=True)



train = my_df[my_df["target"].isnull()==False]

target = train["target"]
from category_encoders import  LeaveOneOutEncoder

leaveOneOut_encoder = LeaveOneOutEncoder()

for nom in ["nom_5","nom_6","nom_9"]:

    train[nom] = leaveOneOut_encoder.fit_transform(train[nom], train["target"])

    test[nom] = leaveOneOut_encoder.transform(test[nom])
train.drop(columns='target', inplace=True)



train.reset_index(drop=True, inplace=True)

test.reset_index(drop=True, inplace=True)

target.reset_index(drop=True, inplace=True)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):

    kf = KFold(n_splits=5)

    fold_splits = kf.split(train, target)

    cv_scores = []

    pred_full_test = 0

    pred_train = np.zeros((train.shape[0]))

    i = 1

    for dev_index, val_index in fold_splits:

        print('Started ' + label + ' fold ' + str(i) + '/5')

        dev_X, val_X = train.loc[dev_index], train.loc[val_index]

        dev_y, val_y = target.loc[dev_index], target.loc[val_index]

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
lr_params = {'solver': 'lbfgs', 'C':0.1, 'max_iter':500}
results = run_cv_model(train, test, target, runLR, lr_params, auc, 'lr')
submission = pd.DataFrame({'id': test_id, 'target': results['test']})

submission.to_csv('submission.csv', index=False)