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
import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head(3)
df.info()
df.describe()
# 전처리 함수 생성

from sklearn.model_selection import train_test_split



def get_preprocessed_df(df=None):

    df_copy = df.copy()

    df_copy.drop('Time', axis=1, inplace=True)

    return df_copy
# 데이터 가공 후 학습/데이터 세트 반환 함수 생성

def get_train_test_dataset(df=None):

    # 전처리 완료

    df_copy = get_preprocessed_df(df)

    x_features = df_copy.iloc[:, :-1]

    y_target = df_copy.iloc[:, -1]

    # target 변수 기준으로 stratify 분리

    x_train, x_test, y_train, y_test = train_test_split(x_features, y_target,

                                                       random_state = 1995,

                                                       test_size = .3,

                                                       stratify = y_target)

    # 학습/테스트 데이터 세트 반환

    return x_train, x_test, y_train, y_test



x_train, x_test, y_train, y_test = get_train_test_dataset(df)
print('학습 데이터 레이블 값 비율')

print(y_train.value_counts() / y_train.shape[0] * 100)

print('테스트 데이터 레이블 값 비율')

print(y_test.value_counts() / y_test.shape[0] * 100)
# 예측 성능 평가 함수 만들기

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve



def get_clf_eval(y_test, pred=None, pred_proba=None):

    confusion = confusion_matrix(y_test, pred)

    accuracy = accuracy_score(y_test, pred)

    precision = precision_score(y_test, pred)

    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)

    

    roc_auc = roc_auc_score(y_test, pred_proba)

    print('오차 행렬 confusion matrix')

    print(confusion)

    print('정확도 accuracy: {0:.4f}, 정밀도 precision: {1:.4f}, 재현율 recall: {2:.4f}, \F1: {3:.4f}, AUC: {4:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
# 모델 만들기 - 로지스틱 회귀

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_pred = lr.predict(x_test)

lr_pred_proba = lr.predict_proba(x_test)[:, 1]



get_clf_eval(y_test, lr_pred, lr_pred_proba)
# 반복적으로 모델을 변경해 학습/예측/평가를 진행하는 함수 생성

def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):

    model.fit(ftr_train, tgt_train)

    pred = model.predict(ftr_test)

    pred_proba = model.predict_proba(ftr_test)[:, 1]

    get_clf_eval(tgt_test, pred, pred_proba)
# 모델 생성 - LightGBM

# 본 데이터 세트를 극도로 불균형한 레이블 값 분포를 가지고 있기 때문에 boost_from_average=False로 지정해야 합니다

# used only in regression, binary, multiclassova and cross-entropy applications

# adjusts initial score to the mean of labels for faster convergence

# 평균치 부스팅으로 하면 값이 맞지 않기 때문에?



from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(n_estimators=1000,

                     num_leaves=64,

                     n_jobs=-1,

                     boost_from_average=False,

                     random_state=1995)

get_model_train_eval(lgbm,

                    ftr_train = x_train,

                    ftr_test = x_test,

                    tgt_train = y_train,

                    tgt_test = y_test)
# 데이터 분포 변환 후 모델 학습/예측/평가

# `Amount` feature 분포도 확인하기



import seaborn as sns

plt.figure(figsize = (8,4))

plt.xticks(range(0, 30000, 1000), rotation=60)

sns.distplot(df['Amount'])
# 정규 분포 스케일링 함수 생성

from sklearn.preprocessing import StandardScaler



def get_preprocessed_df(df=None):

    df_copy = df.copy()

    scaler = StandardScaler()

    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))

    # 변환된 Amount를 Amount_scaled로 피처명 변경 후에 dataframe 맨 앞 column에 입력

    df_copy.insert(0, 'Amount_scaled', amount_n)

    # 기존 time, amount 변수 삭제

    df_copy.drop(columns = ['Time', 'Amount'], inplace=True)

    return df_copy
# amount 스케일링 이후에 로지스틱 회귀 및 lightGBM 수행

x_train, x_test, y_train, y_test = get_train_test_dataset(df)



print('### 로지스틱 회귀 예측 성능 ###')

lr = LogisticRegression()

get_model_train_eval(lr,

                     ftr_train = x_train,

                     ftr_test = x_test,

                     tgt_train = y_train,

                     tgt_test = y_test)

print('### LightGBM 예측 성능 ###')

lgbm = LGBMClassifier(n_estimators = 1000,

                     num_leaves = 64,

                     n_jobs = -1,

                     random_state = 1995,

                     boost_from_average=False)

get_model_train_eval(lgbm,

                    ftr_train = x_train,

                    ftr_test = x_test,

                    tgt_train = y_train,

                    tgt_test = y_test)
# 로그 변환 후 모델 예측/평가

def get_preprocessed_df(df=None):

    df_copy = df.copy()

    # 로그 변환

    amount_n = np.log1p(df_copy['Amount'])

    df_copy.insert(0, 'Amount_scaled', amount_n)

    df_copy.drop(columns = ['Time', 'Amount'], inplace=True)

    return df_copy
x_train, x_test, y_train, y_test = get_train_test_dataset(df)



print('### 로지스틱 회귀 예측 성능 ###')

get_model_train_eval(lr,

                     ftr_train = x_train,

                     ftr_test = x_test,

                     tgt_train = y_train,

                     tgt_test = y_test)

print('### LightGBM 예측 성능 ###')

get_model_train_eval(lgbm,

                    ftr_train = x_train,

                    ftr_test = x_test,

                    tgt_train = y_train,

                    tgt_test = y_test)
# 이상치 제거 후 모델 학습/평가

# dataframe의 correlation 구한 뒤 heatmap으로 나타내기

plt.figure(figsize=(9, 9))

corr = df.corr()

sns.heatmap(corr, cmap='RdBu')
# 이 중 V14에 대해서 이상치를 찾아서 제거하기 

def get_outlier(df=None,

               column=None,

               weight=1.5):

    # fraud에 해당하는 column 데이터만 추출, 1사분위와 3사분위 지점을 구함

    fraud = df[df['Class'] == 1][column]

    quantile_25 = np.percentile(fraud.values, 25)

    quantile_75 = np.percentile(fraud.values, 75)

    # IQR을 구하고 IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함

    iqr = quantile_75 - quantile_25

    iqr_weight = iqr * weight

    lowest_val = quantile_25 - iqr_weight

    highest_val = quantile_75 + iqr_weight

    # 이상치 index 반환

    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index

    return outlier_index
outlier_index = get_outlier(df=df, column='V14', weight=1.5)

print('이상치 데이터 인덱스: ', outlier_index)
# 추출된 이상치 제거 후 모델 적용

def get_preprocessed_df(df=None):

    df_copy = df.copy()

    scaler = StandardScaler()

    amount_n = scaler.fit_transform(df_copy['Amount'].values.reshape(-1, 1))

    df_copy.insert(0, 'Amount_scaled', amount_n)

    df_copy.drop(columns = ['Time', 'Amount'], axis=1, inplace=True)

    # 이상치 삭제 

    outlier_index = get_outlier(df=df_copy, column = 'V14', weight=1.5)

    df_copy.drop(outlier_index, axis=0, inplace=True)

    return df_copy



x_train, x_test, y_train, y_test = get_train_test_dataset(df)



print('### 로지스틱 회귀 예측 성능 ###')

get_model_train_eval(lr,

                     ftr_train = x_train,

                     ftr_test = x_test,

                     tgt_train = y_train,

                     tgt_test = y_test)

print('### LightGBM 예측 성능 ###')

get_model_train_eval(lgbm,

                    ftr_train = x_train,

                    ftr_test = x_test,

                    tgt_train = y_train,

                    tgt_test = y_test)
# SMOTE 오버 샘플링 적용 후 모델 학습/평가

from imblearn.over_sampling import SMOTE



smote = SMOTE(random_state=1995)

x_train_over, y_train_over = smote.fit_sample(x_train, y_train)

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', x_train.shape, y_train.shape)

print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', x_train_over.shape, y_train_over.shape)

print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
# SMOTE 데이터로 로지스틱 회귀 모델 적용

lr = LogisticRegression()

get_model_train_eval(lr,

                    ftr_train = x_train_over, ftr_test = x_test,

                    tgt_train = y_train_over, tgt_test = y_test)
# 임곗값 확인하기

# 임계값 설정

thresholds = [.4, .45, .5, .55, .6]



def precision_recall_curve_plot(y_test, pred_proba_c1):

    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출하기

    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    

    # x축을 threshold값으로, y축은 정밀도, 재현율 값으로 각각 plot 수행. 정밀도는 점선으로 표시

    plt.figure(figsize=(8, 6))

    threshold_boundary = thresholds.shape[0]

    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')

    plt.plot(thresholds, recalls[0:threshold_boundary], label='recall')

    

    # threshold 값 x축의 scale을 0, 1 단위로 변경

    start, end = plt.xlim()

    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    

    # x, y축 label과 legend, grid 설정

    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')

    plt.legend(); plt.grid()

    plt.show()
precision_recall_curve_plot(y_test, lr.predict_proba(x_test)[:, 1])
# SMOTE 데이터로 LGBM 적용

lgbm = LGBMClassifier(random_state = 1995,

                     n_estimators = 1000,

                     num_leaves = 64,

                     n_jobs = -1,

                     boost_from_average = False)

get_model_train_eval(lgbm,

                    ftr_train = x_train_over, ftr_test = x_test,

                    tgt_train = y_train_over, tgt_test = y_test)