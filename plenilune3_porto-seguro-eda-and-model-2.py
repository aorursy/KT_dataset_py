# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# 훈련 데이터, 테스트 데이터를 읽어온다

trn = pd.read_csv('../input/train.csv', na_values=['-1', '-1.0'])

tst = pd.read_csv('../input/test.csv', na_values=['-1', '-1.0'])
# 데이터의 크기를 확인한다

print(trn.shape, tst.shape)
# 데이터 첫 5줄을 확인한다

trn.head()
# 데이터프레임에 대한 메타 정보를 확인한다

trn.info()
# 타겟 변수의 고유값과 타겟==1의 비율을 계산한다

np.unique(trn['target'])

1.0 * sum(trn['target']/trn.shape[0])
# 그 외 기초 통계 기법



# 변수의 최대값, 최소값 등을 확인한다

trn.describe()



# 변수의 결측값을 확인한다

trn.isnull().sum(axis=0)

tst.isnull().sum(axis=0)
# 훈련 데이터와 테스트 데이터를 통합한다

tst['target'] = np.nan

df = pd.concat([trn, tst], axis=0)
# 시각화 관련 라이브러리를 불러온다

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
# 시각화 관련 함수를 미리 정의한다

def bar_plot(col, data, hue=None):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)

    plt.show()

    

def dist_plot(col, data):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.distplot(data[col].dropna(), kde=False, bins=10)

    plt.show()

    

def bar_plot_ci(col, data):

    f, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(x=col, y='target', data=data)

    plt.show()
# 분석의 편의를 위해 변수 유형별로 구분한다

# 이진 변수

binary = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',

          'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_calc_15_bin', 

          'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin']

# 범주형 변수

category = ['ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 

            'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 

            'ps_car_10_cat', 'ps_car_11_cat']

# 정수형 변수

integer = ['ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 

           'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 

           'ps_calc_14', 'ps_car_11']

# 소수형 변수

floats = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_car_12', 'ps_car_13',

          'ps_car_14', 'ps_car_15']
#이진 변수, 범주형 변수 그리고 정수형 변수를 시각화 한다.

for col in binary + category + integer:

    bar_plot(col, df)
# 소수형 변수를 시각화 한다.

for col in floats:

    dist_plot(col, df)
# 전체 데이터에 대한 상관관계 HeatMap 시각화

corr = df.corr()

cmap = sns.color_palette("Blues")

f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corr, cmap=cmap)
# 일부 변수만 선별

features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 

          'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin',

          'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 

          'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 

          'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_14', 'ps_ind_15', 'ps_car_11',

          'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13',

          'ps_car_14', 'ps_car_15']



corr_sub = df[features].corr()

f, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(corr_sub, cmap=cmap)
for col in (binary + category + integer):

    bar_plot_ci(col, df)
df['is_tst'] = df['target'].isnull()

for col in binary + category + integer:

    bar_plot(col, df, 'is_tst')
# 훈련/테스트 데이터를 읽어온다

train = pd.read_csv("../input/train.csv")

train_label = train['target']

train_id = train['id']

del train['target'], train['id']



test = pd.read_csv("../input/test.csv")

test_id = test['id']

del test['id']
# 파생 변수 01 : 결측값을 의미하는 “-1”의 개수를 센다

train['missing'] = (train==-1).sum(axis=1).astype(float)

test['missing'] = (test==-1).sum(axis=1).astype(float)



# 파생 변수 02 : 이진 변수의 합

bin_features = [c for c in train.columns if 'bin' in c]

train['bin_sum'] = train[bin_features].sum(axis=1)

test['bin_sum'] = test[bin_features].sum(axis=1)



# 파생 변수 03 : 단일변수 타겟 비율 분석으로 선정한 변수를 기반으로 Target Encoding을 수행한다. Target Encoding은 교차 검증 과정에서 진행한다.

features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']
# 모델 학습에 필요한 라이브러리

import lightgbm as lgbm

from sklearn.model_selection import StratifiedKFold
# LightGBM 모델의 설정값이다.

num_boost_round = 10000

params = {"objective": "binary",

          "boosting_type": "gbdt",

          "learning_rate": 0.1,

          "num_leaves": 15,

          "max_bin": 256,

          "feature_fraction": 0.6,

          "verbosity": 0,

          "drop_rate": 0.1,

          "is_unbalance": False,

          "max_drop": 50,

          "min_child_samples": 10,

          "min_child_weight": 150,

          "min_split_gain": 0,

          "subsample": 0.9,

          "seed": 2018

}
def Gini(y_true, y_pred):

    # 정답과 예측값의 개수가 동일한지 확인한다

    assert y_true.shape == y_pred.shape

    n_samples = y_true.shape[0]



    # 예측값(y_pred)를 오름차순으로 정렬한다

    arr = np.array([y_true, y_pred]).transpose()

    true_order = arr[arr[:, 0].argsort()][::-1, 0]

    pred_order = arr[arr[:, 1].argsort()][::-1, 0]



    # Lorenz curves를 계산한다

    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)

    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)

    L_ones = np.linspace(1 / n_samples, 1, n_samples)



    # Gini 계수를 계산한다

    G_true = np.sum(L_ones - L_true)

    G_pred = np.sum(L_ones - L_pred)



    # Gini 계수를 정규화한다

    return G_pred * 1. / G_true
# LightGBM 모델 학습 과정에서 평가 함수로 사용한다

def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    return 'gini', Gini(labels, preds), True

# Stratified 5-Fold 내부 교차 검증을 준비한다

NFOLDS = 5

kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)

kf = kfold.split(train, train_label)



cv_train = np.zeros(len(train_label))

cv_pred = np.zeros(len(test_id))    

best_trees = []

fold_scores = []
for i, (train_fold, validate) in enumerate(kf):

    # 훈련/검증 데이터를 분리한다

    X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate, :], train_label[train_fold], train_label[validate]

    

    # target encoding 피쳐 엔지니어링을 수행한다

    for feature in features:

        # 훈련 데이터에서 feature 고유값별 타겟 변수의 평균을 구한다

        map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')

        map_dic = map_dic.to_dict()['target']

        # 훈련/검증/테스트 데이터에 평균값을 매핑한다

        X_train[feature + '_target_enc'] = X_train[feature].apply(lambda x: map_dic.get(x, 0))

        X_validate[feature + '_target_enc'] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))

        test[feature + '_target_enc'] = test[feature].apply(lambda x: map_dic.get(x, 0))



    dtrain = lgbm.Dataset(X_train, label_train)

    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)

    # 훈련 데이터를 학습하고, evalerror() 함수를 통해 검증 데이터에 대한 정규화 Gini 계수 점수를 기준으로 최적의 트리 개수를 찾는다.

    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)

    best_trees.append(bst.best_iteration)

    # 테스트 데이터에 대한 예측값을 cv_pred에 더한다.

    cv_pred += bst.predict(test, num_iteration=bst.best_iteration)

    cv_train[validate] += bst.predict(X_validate)



    # 검증 데이터에 대한 평가 점수를 출력한다.

    score = Gini(label_validate, cv_train[validate])

    print(score)

    fold_scores.append(score)



cv_pred /= NFOLDS
# 시드값별로 교차 검증 점수를 출력한다.

print("cv score:")

print(Gini(train_label, cv_train))

print(fold_scores)

print(best_trees, np.mean(best_trees))
# 테스트 데이터에 대한 결과물을 저장한다.

pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('../lgbm_baseline.csv', index=False)