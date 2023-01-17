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
card=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
card.head()
from sklearn.model_selection import train_test_split



#입력받은 dataframe 복사한뒤 time칼럼 삭제하고 복사한 dataframe 반환 함수

def get_preprocessed_df(df=None):

    df_copy=df.copy()

    df_copy.drop('Time',axis=1,inplace=True)

    return df_copy



def get_train_test_split(df=None):

    df_copy=get_preprocessed_df(df)

    X_features=df_copy.iloc[:,:-1]

    y_target=df_copy.iloc[:,-1]

    X_train,X_test,y_train,y_test=train_test_split(X_features,y_target,test_size=0.3,

                                                  random_state=0,stratify=y_target)

    return X_train,X_test,y_train,y_test



X_train,X_test,y_train,y_test=get_train_test_split(card)
print('train 레이브 비율')

print(y_train.value_counts()/y_train.shape[0]*100,'\n')

print('test 레이블 비율')

print(y_test.value_counts()/y_test.shape[0]*100)
def get_clf_eval(y_test,pred):

    from sklearn.metrics import confusion_matrix

    from sklearn.metrics import roc_auc_score

    from sklearn.metrics import f1_score

    from sklearn.metrics import precision_score

    from sklearn.metrics import recall_score

    from sklearn.metrics import accuracy_score

    confusion=confusion_matrix(y_test,pred)

    accuracy=round(accuracy_score(y_test,pred),4)

    precision=round(precision_score(y_test,pred),4)

    recall=round(recall_score(y_test,pred),4)

    f1=round(f1_score(y_test,pred),4)

    roc_score=round(roc_auc_score(y_test,pred),4)

    

    print('오차행렬')

    print(confusion,'\n')

    print(f'정확도 :{accuracy}, 정밀도: {precision}, 재현율: {recall}, F1 스코어: {f1}, ROC AUC: {roc_score}\n')

    
from sklearn.linear_model import LogisticRegression



lr_clf=LogisticRegression()

lr_clf.fit(X_train,y_train)

lr_pred=lr_clf.predict(X_test)

lr_pred_proba=lr_clf.predict_proba(X_test)[:,1]



get_clf_eval(y_test,lr_pred)
def get_model_train_eval(model,ftr_train=None,ftr_test=None,tgt_train=None,tgt_test=None):

    model.fit(ftr_train,tgt_train)

    pred=model.predict(ftr_test)

    get_clf_eval(tgt_test,pred)
from lightgbm import LGBMClassifier



lgbm_clf=LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1,boost_from_average=False)

get_model_train_eval(lgbm_clf,X_train,X_test,y_train,y_test)
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(8,4))

plt.xticks(range(0,30000,1000),rotation=60)

sns.distplot(card['Amount']);
from sklearn.preprocessing import StandardScaler



def get_preprocessed_df(df=None):

    df_copy=df.copy()

    scaler=StandardScaler()

    amount_n=scaler.fit_transform(df_copy['Amount'].values.reshape(-1,1))

    df_copy.insert(0,'Amount_Scaled',amount_n)

    df_copy.drop(['Time','Amount'],axis=1,inplace=True)

    return df_copy
X_train,X_test,y_train,y_test=get_train_test_split(card)



print('로지스틱 회귀')

lr_clf=LogisticRegression()

get_model_train_eval(lr_clf,X_train,X_test,y_train,y_test)



print('LightGBM')

lgbm_clf=LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1)

get_model_train_eval(lgbm_clf,X_train,X_test,y_train,y_test)
def get_preprocessed_df(df=None):

    df_copy=df.copy()

    amount_n=np.log1p(df_copy['Amount'])

    df_copy.insert(0,'Amount_Scaled',amount_n)

    df_copy.drop(['Time','Amount'],axis=1,inplace=True)

    return df_copy
X_train,X_test,y_train,y_test=get_train_test_split(card)



print('로지스틱 회귀')

lr_clf=LogisticRegression()

get_model_train_eval(lr_clf,X_train,X_test,y_train,y_test)



print('LightGBM')

lgbm_clf=LGBMClassifier(n_estimators=1000,num_leaves=64,n_jobs=-1)

get_model_train_eval(lgbm_clf,X_train,X_test,y_train,y_test)
plt.figure(figsize=(9,9))

corr=card.corr()

sns.heatmap(corr,cmap='RdBu');
def get_outlier(df=None, column=None, weight=1.5):

    # fraud에 해당하는 column 데이터만 추출, 1/4 분위와 3/4 분위 지점을 np.percentile로 구함. 

    fraud = df[df['Class']==1][column]

    quantile_25 = np.percentile(fraud.values, 25)

    quantile_75 = np.percentile(fraud.values, 75)

    # IQR을 구하고, IQR에 1.5를 곱하여 최대값과 최소값 지점 구함. 

    iqr = quantile_75 - quantile_25

    iqr_weight = iqr * weight

    lowest_val = quantile_25 - iqr_weight

    highest_val = quantile_75 + iqr_weight

    # 최대값 보다 크거나, 최소값 보다 작은 값을 아웃라이어로 설정하고 DataFrame index 반환. 

    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index

    return outlier_index

outlier_index = get_outlier(df=card, column='V14', weight=1.5)

print('이상치 데이터 인덱스:', outlier_index)
# get_processed_df( )를 로그 변환 후 V14 피처의 이상치 데이터를 삭제하는 로직으로 변경. 

def get_preprocessed_df(df=None):

    df_copy = df.copy()

    amount_n = np.log1p(df_copy['Amount'])

    df_copy.insert(0, 'Amount_Scaled', amount_n)

    df_copy.drop(['Time','Amount'], axis=1, inplace=True)

    # 이상치 데이터 삭제하는 로직 추가

    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)

    df_copy.drop(outlier_index, axis=0, inplace=True)

    return df_copy



X_train, X_test, y_train, y_test = get_train_test_split(card)

print('### 로지스틱 회귀 예측 성능 ###')

get_model_train_eval(lr_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)

print('### LightGBM 예측 성능 ###')

get_model_train_eval(lgbm_clf, ftr_train=X_train, ftr_test=X_test, tgt_train=y_train, tgt_test=y_test)
from imblearn.over_sampling import SMOTE



smote = SMOTE(random_state=0)

X_train_over, y_train_over = smote.fit_sample(X_train, y_train)

print('SMOTE 적용 전 학습용 피처/레이블 데이터 세트: ', X_train.shape, y_train.shape)

print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트: ', X_train_over.shape, y_train_over.shape)

print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())
lr_clf = LogisticRegression()

# ftr_train과 tgt_train 인자값이 SMOTE 증식된 X_train_over와 y_train_over로 변경됨에 유의

get_model_train_eval(lr_clf, ftr_train=X_train_over, ftr_test=X_test, tgt_train=y_train_over, tgt_test=y_test)
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from sklearn.metrics import precision_recall_curve

%matplotlib inline



def precision_recall_curve_plot(y_test , pred_proba_c1):

    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출. 

    precisions, recalls, thresholds = precision_recall_curve( y_test, pred_proba_c1)

    

    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 Plot 수행. 정밀도는 점선으로 표시

    plt.figure(figsize=(8,6))

    threshold_boundary = thresholds.shape[0]

    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle='--', label='precision')

    plt.plot(thresholds, recalls[0:threshold_boundary],label='recall')

    

    # threshold 값 X 축의 Scale을 0.1 단위로 변경

    start, end = plt.xlim()

    plt.xticks(np.round(np.arange(start, end, 0.1),2))

    

    # x축, y축 label과 legend, 그리고 grid 설정

    plt.xlabel('Threshold value'); plt.ylabel('Precision and Recall value')

    plt.legend(); plt.grid()

    plt.show()
precision_recall_curve_plot( y_test, lr_clf.predict_proba(X_test)[:, 1] )
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False)

get_model_train_eval(lgbm_clf, ftr_train=X_train_over, ftr_test=X_test,

                  tgt_train=y_train_over, tgt_test=y_test)