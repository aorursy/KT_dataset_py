# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgbm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# load data

train = pd.read_csv('../input/train.csv')

train_label = train['target']

train_id = train['id']

del train['target'], train['id']



test = pd.read_csv('../input/test.csv')

test_id = test['id']

del test['id']
print(train.shape)

print(test.shape)
train.head()
train.info()
print(train_label.unique())

print('target 0:', len(train_label[train_label==0])/len(train) * 100)

print('target 1:', len(train_label[train_label==1])/len(train) * 100)
def bar_plot(col, data, hue=None):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)

    plt.show()

    

def dist_plot(col, data):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(data[col].dropna(), kde=False, bins=10)

    plt.show()
# binary variables

binary = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',

          'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 

          'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']

# categorical variables

category = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 

            'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 

            'ps_car_10_cat', 'ps_car_11_cat']

# integer variables

integer = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 

           'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 

           'ps_calc_14', 'ps_car_11']

# floats variables

floats = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_car_12', 'ps_car_13',

          'ps_car_14', 'ps_car_15']
# merge train & test data

df = pd.concat([train, test], axis=0)
# ploting binary, category, integer variables

for col in binary+category+integer:

    bar_plot(col, df)
# ploting float variables

for col in floats:

    dist_plot(col, df)
corr = df.corr()



cmap = sns.color_palette('Blues')

f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corr, cmap=cmap)
# 파생변수 1: 결측값을 의미하는 '-1'의 개수

train['missing'] = (train==-1).sum(axis=1).astype(float)

test['missing'] = (test==-1).sum(axis=1).astype(float)



# 파생변수 2: 이진 변수의 합

bin_features = [c for c in train.columns if 'bin' in c]

train['bin_sum'] = train[bin_features].sum(axis=1)

test['bin_sum'] = test[bin_features].sum(axis=1)



# 파생변수 3: target encoding

features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 

            'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 

            'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 

            'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']
def Gini(y_true, y_pred):

    # check and get number of samples

    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]

    

    # sort rows on prediction column 

    # (from largest to smallest)

    arr = np.array([y_true, y_pred]).transpose()

    true_order = arr[arr[:,0].argsort()][::-1,0]

    pred_order = arr[arr[:,1].argsort()][::-1,0]

    

    # get Lorenz curves

    L_true = np.cumsum(true_order) / np.sum(true_order)

    L_pred = np.cumsum(pred_order) / np.sum(pred_order)

    L_ones = np.linspace(1/n_samples, 1, n_samples)

    

    # get Gini coefficients (area between curves)

    G_true = np.sum(L_ones - L_true)

    G_pred = np.sum(L_ones - L_pred)

    

    # normalize to true Gini coefficient

    return G_pred/G_true



def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'gini', Gini(labels, preds), True
# Parameters of LightGBM

num_boost_round = 10000

params = {

    'objective':'binary',

    'boosting_type':'gbdt',

    'learning_rate':0.1,

    'num_leaves':15,

    'max_bin':256,

    'feature_fraction':0.6,

    'verbosity':0,

    'drop_rate':0.1,

    'is_unbalance':False,

    'max_drop':50,

    'min_child_samples':10,

    'min_child_weight':150,

    'min_split_gain':0,

    'subsample':0.9,

    'seed':2018

}
# Model Training & Cross Validation

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

kf = kfold.split(train, train_label)



cv_train = np.zeros(len(train_label))

cv_pred = np.zeros(len(test_id))

best_trees = []

fold_scores = []



for i, (train_fold, validate) in enumerate(kf):

    # Split train/validate

    X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate,:], train_label[train_fold], train_label[validate]

    

    # target encoding

    for feature in features:

        # 훈련 데이터에서 feature 고유값별로 타겟 변수의 평균을 구함 

        map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')

        map_dic = map_dic.to_dict()['target']

        

        X_train[feature+'_target_enc'] = X_train[feature].apply(lambda x: map_dic.get(x, 0))

        X_validate[feature+'_target_enc'] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))

        test[feature+'_target_enc'] = test[feature].apply(lambda x: map_dic.get(x, 0))

        

    dtrain = lgbm.Dataset(X_train, label_train)

    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)

    

    # evalerror()를 통해 검증 데이터에 대한 정규화 Gini계수 점수를 기준으로 한 최적의 트리 개수

    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, 

                    verbose_eval=100, early_stopping_rounds=100)

    best_trees.append(bst.best_iteration)

    

    # predict

    cv_pred += bst.predict(test, num_iteration=bst.best_iteration)

    cv_train[validate] += bst.predict(X_validate)

    

    # score

    score = Gini(label_validate, cv_train[validate])

    print(score)

    fold_scores.append(score)

    

cv_pred /= NFOLDS
print("cv score:")

print(Gini(train_label, cv_train))

print(fold_scores)

print(best_trees, np.mean(best_trees))
# save prediction

pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('submission.csv', index=False)