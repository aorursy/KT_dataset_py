# 필요환경 구성



from __future__ import print_function

import os



data_path = ['../input/intelml101class1']
import numpy as np

import pandas as pd



filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])

#print(filepath);

data = pd.read_csv(filepath)



# 사례 고객: 5,000명

data.tail(n=1).T
# 기본적 통계수치 확인

data.describe().T
# 필요없는 열 삭제

data_raw = data.copy() # 원형 데이터 저장 (의도치 않게 data 변수가 변경될 경우 대비)



col_unused = ['account_length', 'area_code', 'phone_number']

data_slim = data.drop(columns=col_unused)

#print(len(data_slim.columns)) # 18 = 21 - 3

data_slim.tail(n=1).T
import pandas as pd



# 데이터 불러오기

filepath = os.sep.join(data_path + ['Orange_Telecom_Churn_Data.csv'])

data = pd.read_csv(filepath)
data.head(1).T
# 필요없는 열 삭제

data.drop(['state', 'area_code', 'phone_number'], axis=1, inplace=True)

data.columns
# 전체 열 데이터 종류 확인

pd.DataFrame(data_slim).dtypes
# 실험적으로 state부터 변환 / factorize 활용



# 변환 전

print(data_slim.state.tail(n=3), '\n', '\n', 'state 유일값 수: ', len(data_slim.state.unique()), '\n', data_slim.state.unique())
# 변환 후

data_slim.state= pd.factorize(data_slim.state)[0]

print(data_slim.state.tail(n=3), '\n', '\n', 'state 유일값 수: ', len(data_slim.state.unique()), '\n', data_slim.state.unique())
# 나머지 비숫자 열 변환

for i in range(len(data_slim.columns)):

    if data_slim.dtypes[i] != int and data_slim.dtypes[i] != float: # 숫자 여부 확인

        col = data_slim.columns[i]

        data_slim[col] = pd.factorize(data_slim[col])[0]

data_slim.tail(n=3).T # 변환된 내용
# 변환 결과 (열 종류)

data_slim.dtypes
# 추가적으로, 숫자지만 범주에 가까운 열 (number_customer_service_calls) 확인



# 변환 전

print('변환 전 -', data_slim.number_customer_service_calls.unique())



# 변환 가능해보임 > 해보자 > 실은 변환할 필요없었음 > 이미 숫자임!

data_slim['number_customer_service_calls'] = pd.factorize(data_slim.number_customer_service_calls)[0]



# 변환 후

print('변환 후 -', data_slim.number_customer_service_calls.unique())
import matplotlib.pyplot as plt



plt.style.use('seaborn-dark')

plt.xlabel('No. of Customer Service Calls')

plt.ylabel('Frequency')

plt.title('Frequency by No. of Service Calls (raw data)')

plt.hist(data_raw.number_customer_service_calls, color='k', alpha=0.67);
serv_call_fact_sort = pd.factorize(data_raw.number_customer_service_calls, sort=True)[0]

print('서비스콜 수 기준, 원형 데이터와 일치하는 고객 수:', sum(serv_call_fact_sort == data_raw.number_customer_service_calls))

print('서비스콜 수 총합:', sum(serv_call_fact_sort))

plt.style.use('seaborn-dark')

plt.xlabel('No. of Customer Service Calls')

plt.ylabel('Frequency')

plt.title('Frequency by No. of Service Calls (sorted factorized)')

plt.hist(serv_call_fact_sort, color='navy', alpha=0.67);
serv_call_fact_unsort = pd.factorize(data_raw.number_customer_service_calls)[0] #false가 기본값

print('서비스콜 수 기준, 원형 데이터와 일치하는 고객 수:', sum(serv_call_fact_unsort == data_raw.number_customer_service_calls))

print('서비스콜 수 총합:', sum(serv_call_fact_unsort))

plt.style.use('seaborn-dark')

plt.xlabel('No. of Customer Service Calls')

plt.ylabel('Frequency')

plt.title('Frequency by No. of Service Calls (unsorted factorized)')

plt.hist(serv_call_fact_unsort, color='crimson', alpha=0.67);
# 지표간 상관관계 / 지표별 고객이탈율과의 상관관계



import seaborn as sns



sns.set(style="dark")



# 상관관계 매트릭스

corr = data_raw.corr()



# 중복되는 윗부분 가리기

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(7, 5))

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# 상관관계 그래프

corr_mat = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

                       square=True, linewidths=.5, cbar_kws={"shrink": .5})
# sorted된 factorizer로 업데이트

data_slim['number_customer_service_calls'] = serv_call_fact_sort
# StandardScaler

from sklearn.preprocessing import StandardScaler



"""

잠깐 메모: 파이썬식 변수 이름 짓기 (검색 후 적용)

> 계속 아래_막대_방법을 적용하자,

  작업하고 있는 환경이 엄격하게 camelCase를 따르는 게 아니라면

"""

StdSc = StandardScaler()

data_slim_std_scaled = pd.DataFrame(StdSc.fit_transform(data_slim), columns = data_slim.columns)



data_slim_std_scaled.describe().T.tail(n=3)
# 타겟 확인 (스케일링 전) > 0/1로 분류

print('스케일링 전 타겟')

data_slim.churned.tail(n=5)
# 타겟 확인 (스케일링 후) > 큰일 남!

print('스케일링 후 타겟')

data_slim_std_scaled.churned.tail(n=5)
# MinMaxScaler (기본 범주인 (0,1) 적용)

from sklearn.preprocessing import MinMaxScaler



#MMSc = MinMaxScaler(feature_range=(0,3)) # 스케일링 범주에 따른 정확도 차이 확인용

MMSc = MinMaxScaler()

data_slim_MMSc_scaled = pd.DataFrame(MMSc.fit_transform(data_slim), columns = data_slim.columns)



data_slim_MMSc_scaled.describe().T.tail(n=3)
# MaxAbsScaler

from sklearn.preprocessing import MaxAbsScaler



MASc = MaxAbsScaler()

data_slim_MASc_scaled = pd.DataFrame(MASc.fit_transform(data_slim), columns = data_slim.columns)



data_slim_MASc_scaled.describe().T.tail(n=3)
# 수치화 후, 스케일링 전 데이터 min 확인: 모든 열 >= 0

pd.DataFrame(data_slim.describe().min(), columns=['min'])
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()



for col in ['intl_plan', 'voice_mail_plan', 'churned']:

    data[col] = lb.fit_transform(data[col])
# sklearn 경고 끄기

import warnings

warnings.filterwarnings('ignore', module='sklearn')



from sklearn.preprocessing import MinMaxScaler



msc = MinMaxScaler()



data = pd.DataFrame(msc.fit_transform(data),

                    columns=data.columns)

data.describe().T.tail(n=3)
# 실험용 데이터 복사

data_lb = data_slim.copy()



from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()



for column in ['intl_plan', 'voice_mail_plan', 'churned']:

    data_lb[column] = lb.fit_transform(data[column])



data_lb.tail(n=1).T
# 타겟 변환 여부 상세 확인

data_lb.churned[-5:]
# 위에서 배운 것(drop) 적용

y_MMSc = data_slim_MMSc_scaled['churned'] # 이름 짧게

X_MMSc = data_slim_MMSc_scaled.drop(columns='churned')

X_MMSc.tail(n=3).T.tail(n=3)
print(y_MMSc.tail(n=3).T)
from sklearn.neighbors import KNeighborsClassifier



KNN = KNeighborsClassifier(n_neighbors=3)

KNN = KNN.fit(X_MMSc, y_MMSc)



y_predict = KNN.predict(X_MMSc)

print('KNN 학습 후 예측값:', y_predict)



# 한 뼘 더: KNN에 탑재된 score 기능을 활용해서 정확도 측정

st_accuracy = KNN.score(X_MMSc, y_MMSc)

print('예측 정확도:', st_accuracy)
# 타겟을 제외한 열 목록

x_cols = [x for x in data.columns if x != 'churned']



# 데이터 두 개로 나누기

X_data = data[x_cols]

y_data = data['churned']



# 다른 방법:

# X_data = data.copy()

# y_data = X_data.pop('churned')
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)



knn = knn.fit(X_data, y_data)



y_pred = knn.predict(X_data)

y_pred
popped_X = data.copy()

popped_y = popped_X.pop('churned')

popped_X.tail(n=3).T.tail(n=3)
print(popped_y.tail(n=3))
# 정확도 측정기

def get_accuracy(prediction, target):

    return sum(prediction == target) / float(len(prediction))



print('우리 정확도:', get_accuracy(y_predict, y_MMSc))
# 맞는 예측 비율 계산기

def accuracy(real, predict):

    return sum(real == predict) / float(real.shape[0])
print('Intel 정확도:', accuracy(y_data, y_pred))
print('Intel 정확도:', get_accuracy(y_data, y_pred), '   우리 정확도:', get_accuracy(y_MMSc, y_predict))
print('인텔 예측과 우리 예측의 일치율:', get_accuracy(y_pred, y_predict))
print('인텔 예측:', sum(y_pred), '  우리 예측:', sum(y_predict)) # 예측값

print('인텔 실제:', sum(y_data), '  우리 실제:', sum(y_MMSc)) # 실제값
print('인텔 예측 일치 고객 수: ', sum(y_pred==y_data), '   우리 예측 일치 고객 수: ', sum(y_predict==y_MMSc))
print('account_length 유일값 수: ', len(data.account_length.unique()), '   커버리지: ', len(data)/len(data.account_length.unique()))
print('state 유일값 수: ', len(data_slim.state.unique()), '   커버리지: ', len(data_slim)/len(data_slim.state.unique()))
# 새 데이터 준비

ohe_st_data = data_raw.copy()

ohe_st_data.drop(['account_length', 'area_code', 'phone_number'], axis=1, inplace=True)

ohe_st_data.tail(n=3).T;



# 각 주별 0/1로 구분하는 열 얻기

st_encoded = pd.get_dummies(ohe_st_data.state, prefix=['state'], drop_first=True)

#print(data_raw.state.nunique(), len(st_encoded.columns)) # 51개 주를 위한 50가지 1/0 구분

st_encoded.describe().T.tail(n=3);



# 인코딩 안 된 다른 열들과 합치기

ohe_st_data = pd.concat([ohe_st_data, st_encoded], axis=1)

#print(len(ohe_st_data.columns))



# 원래 state열 삭제

ohe_st_data.drop(['state'],axis=1, inplace=True)

#print(len(ohe_st_data.columns))

ohe_st_data.tail(n=3).T;



# LB로 Y/N열 변환

lb = LabelBinarizer()

for col in ['intl_plan', 'voice_mail_plan', 'churned']:

    ohe_st_data[col] = lb.fit_transform(ohe_st_data[col])

    

# MinMaxScaler로 스케일링 (기본 범주[0,1] 적용)

MMs = MinMaxScaler()

ohe_st_data_mms = pd.DataFrame(MMs.fit_transform(ohe_st_data), columns = ohe_st_data.columns)

ohe_st_data_mms.describe().T.head(n=3);



# pop으로 X,y 얻기

ohe_st_X = ohe_st_data_mms.copy()

ohe_st_y = ohe_st_X.pop('churned')



# KNN 활용하여 학습 후 정확도 재측정

ohe_st_KNN = KNeighborsClassifier(n_neighbors=3)

ohe_st_KNN = ohe_st_KNN.fit(ohe_st_X, ohe_st_y)

ohe_st_pred = ohe_st_KNN.predict(ohe_st_X)

#print('Predicted Values: ', ohe_st_pred)

ohe_st_accu = ohe_st_KNN.score(ohe_st_X, ohe_st_y)

print('예측정확도:', ohe_st_accu)

print('실제값과 일치하는 예측값의 수: ', sum(ohe_st_pred==ohe_st_y))
# 각 지표가 갖는 유일값 확인 (처음 저장해둔 원형 데이터 활용)

print('area code 유일값:', data_raw.area_code.unique())

print('state 유일값:', data_raw.state.unique())



# 두 수치 비교

area_coverage, state_coverage = len(data_raw)/len(data_raw.area_code.unique()), len(data_raw)/len(data_raw.state.unique())

print('두 지표의 각 값이 포함하는 고객 수 비율: ', area_coverage/state_coverage)
# 1. 필요없는 열 삭제

ac_col_unused = ['state', 'account_length', 'phone_number']

ac_data = data_raw.drop(columns=ac_col_unused)

#print(len(ac_data.columns)) # 18 = 21 - 3

ac_data.tail(n=1);



# 2. LabelBinarizer로 변환

lb = LabelBinarizer()

for i in range(len(ac_data.columns)):

    if ac_data.dtypes[i] != int and ac_data.dtypes[i] != float: # 숫자 여부 확인

        col = ac_data.columns[i]

        ac_data[col]= lb.fit_transform(ac_data[column])

ac_data.tail(n=1).T;



# 3. MinMaxScaler로 스케일링 (기본 범주[0,1] 적용)

MMs = MinMaxScaler()

ac_data_MMs = pd.DataFrame(MMs.fit_transform(ac_data), columns = ac_data.columns)

ac_data_MMs.describe().T.head(n=3);



# 4. pop으로 X,y 얻기

ac_X = ac_data_MMs.copy()

ac_y = ac_X.pop('churned')

#print(ac_y.tail(n=3))

ac_X.tail(n=3).T.tail(n=3);



# 5. KNN 학습 / 이탈 여부 예측

ac_KNN = KNeighborsClassifier(n_neighbors=3)

ac_KNN = ac_KNN.fit(ac_X, ac_y)

ac_pred = ac_KNN.predict(ac_X)

#print(ac_pred)

ac_accu_k3 = ac_KNN.score(ac_X, ac_y)

print('예측정확도:', ac_accu_k3)
# 세 가지 측정기로 area_code 정확도 재확인



ac_accu_my = get_accuracy(ac_y, ac_pred)

ac_accu_intel = accuracy(ac_y, ac_pred)

ac_accu_given = ac_KNN.score(ac_X, ac_y)



print('우리 측정기: ', ac_accu_my)

print('인텔 측정기: ', ac_accu_intel)

print('KNN.score: ', ac_accu_given)

# 실제값/예측값 일치 정도

print('실제값과 일치하는 예측값의 수:', sum(ac_y==ac_pred))
# k값 증가에 따른 정확도 기록기 (우리 측정기 활용)

def accu_lister(X, y, k):

    list = []

    for i in range(k, k+50): 

        KNN = KNeighborsClassifier(n_neighbors=i+1)

        KNN = KNN.fit(X, y)

        pred = KNN.predict(X)

        list.append(get_accuracy(y, pred))

    return list

k = 1380

accu_list = accu_lister(ac_X, ac_y, k);



# K값 증가에 따른 정확도 변화 그래프

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

plt.style.use('seaborn-dark')

plt.xlabel('K')

plt.ylabel('Accuracy')

plt.title('Change of Accuracy by Increase of K')

plt.plot(np.arange(k, k+50, step=1), accu_list, color='k', alpha=0.73, linewidth=7.3);
ac_KNN = KNeighborsClassifier(n_neighbors=1420)

ac_KNN = ac_KNN.fit(ac_X, ac_y)



ac_pred = ac_KNN.predict(ac_X)

ac_accu = ac_KNN.score(ac_X, ac_y)

print('k=1420일 때 예측정확도:', ac_accu)

#print(ac_pred)

print('모든 (이탈)예측의 합:', sum(ac_pred))
# 트레이닝셋/테스트셋으로 데이터 나누기

def data_split(X, y, s, r): # s=전체 데이터 중 테스트셋 비율 / r=랜덤시드

    from sklearn.model_selection import train_test_split

    len_list = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s, random_state=r)

    for i in [X_train, X_test, y_train, y_test]:

        len_list.append(len(i))

    return X_train, X_test, y_train, y_test, len_list

ac_X_train, ac_X_test, ac_y_train, ac_y_test, ac_len_list = data_split(ac_X, ac_y, 0.3, 13) # area_code

st_X_train, st_X_test, st_y_train, st_y_test, st_len_list = data_split(X_MMSc, y_MMSc, 0.3, 13) # state

al_X_train, al_X_test, al_y_train, al_y_test, al_len_list = data_split(X_data, y_data, 0.3, 13); # account_length
# 이 중 area_code 트레이닝셋 확인

ac_X_train.head(n=3).T.head(n=3)
# 이 중 area_code 테스트셋 확인

ac_X_test.head(n=3).T.head(n=3)
# logistic regression으로 학습

def lr_fit(X_train, y_train, X_test, y_test, penalty, C):

    from sklearn.linear_model import LogisticRegression

    LR= LogisticRegression(penalty=penalty, C=C) # penalty = 학습제한 방법 / C = 제한 정도 (클수록 덜 제한)

    LR= LR.fit(X_train, y_train)

    predict = LR.predict(X_test)

    accuracy = get_accuracy(y_test, predict)

    return predict, accuracy



lr_ac_pred, lr_ac_accu = lr_fit(ac_X_train, ac_y_train, ac_X_test, ac_y_test, 'l2', 0.01)

lr_st_pred, lr_st_accu = lr_fit(st_X_train, st_y_train, st_X_test, st_y_test, 'l2', 0.01)

lr_al_pred, lr_al_accu = lr_fit(al_X_train, al_y_train, al_X_test, al_y_test, 'l2', 0.01)

result = {'Area Code': [lr_ac_accu, sum(lr_ac_pred)],

          'State': [lr_st_accu, sum(lr_st_pred)],

         'Account Length': [lr_al_accu, sum(lr_al_pred)]}

result = pd.DataFrame(result, index=['Accuracy', 'Sum of predictions'])

result
# Logistic Rregression에 탑재된 cross validation으로 최적값 학습 후 다시 예측

def LR_cross_valid(X_train, y_train, X_test, y_test, f, r): # f = 트레이닝 데이터 분리 수 / r = 랜덤시드

    from sklearn.linear_model import LogisticRegressionCV

    LR_CV = LogisticRegressionCV(cv=f, random_state=r).fit(X_train, y_train)

    LR_CV.predict(X_test)

    prediction = LR_CV.predict(X_test)

    accuracy = get_accuracy(prediction, y_test)

    return prediction, accuracy



lr_ac_pred_cv, lr_ac_accu_cv = LR_cross_valid(ac_X_train, ac_y_train, ac_X_test, ac_y_test, 5, 13)

lr_st_pred_cv, lr_st_accu_cv = LR_cross_valid(st_X_train, st_y_train, st_X_test, st_y_test, 5, 13)

lr_al_pred_cv, lr_al_accu_cv = LR_cross_valid(al_X_train, al_y_train, al_X_test, al_y_test, 5, 13)

result_cv = {'Area Code': [lr_ac_accu_cv, sum(lr_ac_pred_cv)],

          'State': [lr_st_accu_cv, sum(lr_st_pred_cv)],

         'Account Length': [lr_al_accu_cv, sum(lr_al_pred_cv)]}

#print('Results from LR (with parameters picked by LogisticRegressionCV)')

result_cv = pd.DataFrame(result_cv, index=['Accuracy', 'Sum of predictions'])

result_cv
# encoding된 state 데이터셋 나누기

ohe_st_X_train, ohe_st_X_test, ohe_st_y_train, ohe_st_y_test, ohe_st_len_list = data_split(ohe_st_X, ohe_st_y, 0.3, 13) # state 활용



# 학습제한값 수동 적용한 Logistic Regression으로 학습

ohe_st_pred_lr, ohe_st_accu_lr = lr_fit(ohe_st_X_train, ohe_st_y_train, ohe_st_X_test, ohe_st_y_test, 'l2', 0.01)



# Cross Validation으로 얻은 학습제한값 적용한 Logistic Regression으로 학습

ohe_st_pred_cv, ohe_st_accu_cv = LR_cross_valid(ohe_st_X_train, ohe_st_y_train, ohe_st_X_test, ohe_st_y_test, 5, 13)



print('수동 C 적용 Logistic Regression 예측정확도(state, non-ordinal):', ohe_st_accu_lr)

print('CV C 적용 Logistic Regression 예측정확도(state, non-ordinal):', ohe_st_accu_cv)
# scaling 전 area_code 포함된 데이터 불러오기

ohe_ac_data = ac_data.copy()



# 각 area_code별 0/1로 구분하는 열 얻기

ac_encoded = pd.get_dummies(ohe_ac_data.area_code, prefix=['area_code'], drop_first=True)

#print(data_raw.area_code.nunique(), len(ac_encoded.columns)) # 3개 area_code를 위한 2가지 1/0 구분

ac_encoded.describe().T.tail(n=3);



# 인코딩 안 된 다른 열들과 합치기

ohe_ac_data = pd.concat([ohe_ac_data, ac_encoded], axis=1)

#print(len(ohe_ac_data.columns)) # 20 = 18 + 2



# 원래 area_code열 삭제

ohe_ac_data.drop(['area_code'],axis=1, inplace=True)

#print(len(ohe_ac_data.columns)) # 19 = 20 - 1

ohe_ac_data.tail(n=3).T;



# MinMaxScaler로 스케일링 (기본 범주[0,1] 적용)

MMs = MinMaxScaler()

ohe_ac_data_mms = pd.DataFrame(MMs.fit_transform(ohe_ac_data), columns = ohe_ac_data.columns)

ohe_ac_data_mms.describe().T.tail(n=3);



# pop으로 X,y 얻기

ohe_ac_X = ohe_ac_data_mms.copy()

ohe_ac_y = ohe_ac_X.pop('churned')



# KNN 활용하여 학습 후 정확도 재측정

ohe_ac_KNN = KNeighborsClassifier(n_neighbors=3)

ohe_ac_KNN = ohe_ac_KNN.fit(ohe_ac_X, ohe_ac_y)

ohe_ac_pred = ohe_ac_KNN.predict(ohe_ac_X)

#print(ohe_ac_pred)

#print(sum(ohe_ac_pred==ohe_ac_y))

ohe_ac_accu = ohe_ac_KNN.score(ohe_ac_X, ohe_ac_y)

print('KNN 예측정확도:',ohe_ac_accu)





# Logistic Regression 활용하여 학습 후 정확도 재측정

# 트레이닝셋 / 테스트셋 나누기

ohe_ac_X_train, ohe_ac_X_test, ohe_ac_y_train, ohe_ac_y_test, _ = data_split(ohe_ac_X, ohe_ac_y, 0.3, 13);

# 학습제한값 수동 적용 logistic regression으로 학습

ohe_ac_pred_lr, ohe_ac_accu_lr = lr_fit(ohe_ac_X_train, ohe_ac_y_train, ohe_ac_X_test, ohe_ac_y_test, 'l2', 0.01)

# Cross validation으로 학습된 제한값 적용 logistic regression으로 학습

ohe_ac_pred_cv, ohe_ac_accu_cv = LR_cross_valid(ohe_ac_X_train, ohe_ac_y_train, ohe_ac_X_test, ohe_ac_y_test, 5, 13)

print('LR 예측정확도:',ohe_ac_accu_lr)

print('LR_CV 예측정확도:',ohe_ac_accu_cv)
import pandas as pd

accuracies = pd.DataFrame({'feature': ['account_length', 'state', 'state', 'area_code', 'area_code'], 

                          'encoding': ['not applicable', 'ordinal', 'non-ordinal', 'ordinal', 'non-ordinal'],

                          'KNN': [0.9422, 0.9434, ohe_st_accu, ac_accu_k3, ohe_ac_accu], 

                          'LR': [lr_al_accu, lr_st_accu, ohe_st_accu_lr, lr_ac_accu, ohe_ac_accu_lr],

                          'LR-CV': [lr_al_accu_cv, lr_st_accu_cv, ohe_st_accu_cv, lr_ac_accu_cv, ohe_ac_accu_cv]},

                          columns=['feature', 'encoding', 'KNN', 'LR', 'LR-CV'])

#accuracies.set_index(['feature', 'encoding'])

accuracies
# 전체 예측 중 이탈 고객 비율

print('전체 고객 중 이탈 고객의 비율:', sum(y_data)/len(y_data))
# area_code 열과 해당 고객이탈 열만으로 구성된 데이터 준비

data_ac_only = pd.DataFrame(data_raw.area_code).join(y_data)



# 각 area_code의 총 이탈 고객 수 데이터에 추가

data_ac_only_sum = data_ac_only.groupby('area_code')['churned'].sum().to_frame().reset_index()



# 각 area_code의 총 고객 수 데이터에 추가

data_ac_only_sum['total'] = data_raw.area_code.value_counts().tolist()



# 각 area_code의 총 고객 중 이탈한 고객의 비율 데이터에 추가

data_ac_only_sum['propotion']=data_ac_only_sum.churned/data_ac_only_sum.total

data_ac_only_sum
# 기본 수치 확인 - 218개 유일값 / 평균 100 / 1부터 243까지

print('acount_length 유일값의 수:', data_raw.account_length.nunique())
data_raw.account_length.describe()
# 빈도수 확인

plt.figure(figsize=(7, 5))

plt.style.use('seaborn-dark')

plt.xlabel('Account Length')

plt.ylabel('Frequency')

plt.title('Number of Customers by Account Length')

plt.hist(data_raw.account_length, color='darkcyan');
# 데이터 불러오기

data_telco = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



# 1:1 대조 (account_length 단위 월(30일로 가정)로 변경)

comparison = pd.DataFrame(data_telco.tenure.describe()).join(pd.DataFrame(data_raw.account_length.describe()))

comparison.iloc[1:,1] = comparison.iloc[1:,1]/4.5

comparison
# 히스토그램

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5))



plt.subplot(1, 2, 1)

plt.hist(data_telco.tenure, color='darkcyan', alpha=0.37)

plt.title('Tenure (Telco)')



plt.subplot(1, 2, 2)

plt.hist(data_raw.account_length/4.5, color='darkcyan')

plt.title('Account Length (Orange Telecom)')



fig.tight_layout()
# 테스트셋 비율에 따른 정확도 측정기

def ratio_acc(ratio_X, ratio_y, ratio_list):

    lr_accu_list = []

    cv_accu_list = []

    ratio_check = []

    for ratio in ratio_list:

        ratio_X_train, ratio_X_test, ratio_y_train,  ratio_y_test, _ = data_split(ratio_X, ratio_y, ratio, 13) # 랜덤시드=13

        ratio_pred_lr, ratio_accu_lr = lr_fit(ratio_X_train, ratio_y_train, ratio_X_test, ratio_y_test, 'l2', 0.01)

        ratio_pred_cv, ratio_accu_cv = LR_cross_valid(ratio_X_train, ratio_y_train, ratio_X_test,  ratio_y_test, 5, 13) # cross validation용 데이터 분리 수=5, 랜덤시드 =13

        lr_accu_list.append(ratio_accu_lr)

        cv_accu_list.append(ratio_accu_cv)

        ratio_check.append(ratio)

    return lr_accu_list, cv_accu_list, ratio_check



# 두 버전의 logistic regression(수동 제한값 / CV 제한값)으로 정확도 측정

ratio_list = [0.1, 0.3, 0.5, 0.7, 0.9]

lr_ratio_accu_list, cv_ratio_accu_list, ratio_check = ratio_acc(ohe_ac_X, ohe_ac_y, ratio_list)



# 테스트셋 비율 증가에 따른 정확도 변화 그래프

plt.figure(figsize=(7, 5))

plt.style.use('seaborn-dark')

plt.xlabel('The ratio of Testset')

plt.ylabel('Accuracy')

plt.title('Change of Accuracy by Increase of Testset Ratio')

plt.plot(ratio_check, lr_ratio_accu_list, color='crimson', alpha=0.73, linewidth=3, label="LR")

plt.plot(ratio_check, cv_ratio_accu_list, color='navy', alpha=0.73, linewidth=3, label="LR-CV")

plt.legend(["LR", "LR-CV"], loc="best");
# 최종 모델 학습기

def final_model(X_train, y_train, X_test, y_test, f, r): # f = 트레이닝 데이터 분리 수 / r = 랜덤시드

    from sklearn.linear_model import LogisticRegressionCV

    LR_CV = LogisticRegressionCV(cv=f, random_state=r).fit(X_train, y_train)

    prediction = LR_CV.predict(X_test)

    accuracy = get_accuracy(prediction, y_test)

    return LR_CV, prediction, accuracy



# 최종 모델 얻기 / 해당 모델로 예측정확도 재확인

final_X_train, final_X_test, final_y_train, final_y_test, _ = data_split(ohe_ac_X, ohe_ac_y, 0.3, 13) # 테스트셋 비율=30%, 랜덤시드=13

final_model, final_pred_cv, final_accu_cv = final_model(final_X_train, final_y_train, final_X_test,  final_y_test, 5, 13) # cross validation용 데이터 분리 수=5, 랜덤시드 =13



print('최종 모델을 활용한 고객이탈율 예측정확도: ', final_accu_cv*100, '%')
from __future__ import print_function # 출력 기능은 셀 첫 줄에 위치



# 최초 데이터 입력에서 최종 이탈 예측까지 이어지는 파이프라인

def data_to_predict(data, model=final_model, real=0):

    ## 0. 필요환경 구성 / 데이터 불러오기

    import os

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    import seaborn as sns

    import warnings

    warnings.filterwarnings('ignore', module='sklearn')

    from sklearn.preprocessing import StandardScaler

    from sklearn.preprocessing import MinMaxScaler

    from sklearn.preprocessing import MaxAbsScaler

    from sklearn.preprocessing import LabelBinarizer

    from sklearn.neighbors import KNeighborsClassifier

    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LogisticRegression

    from sklearn.linear_model import LogisticRegressionCV

    data_path = ['../input/intelml101class1']

    data = pd.read_csv(filepath)



    ## 1. 데이터 전처리

    data_new = data.copy() # 비상용 원본 저장

    ac_col_unused = ['state', 'account_length', 'phone_number']

    ac_data = data_new.drop(columns=ac_col_unused)

    if real == 1:

        for column in ['intl_plan', 'voice_mail_plan']:

            ac_data[column] = lb.fit_transform(ac_data[column])

    else:

        for column in ['intl_plan', 'voice_mail_plan', 'churned']:

            ac_data[column] = lb.fit_transform(ac_data[column])

    ac_encoded = pd.get_dummies(ac_data.area_code, prefix=['area_code'], drop_first=True)

    all_ac_data = pd.concat([ac_data, ac_encoded], axis=1)

    all_ac_data.drop(['area_code'], axis=1, inplace=True)

    all_ac_data_mms = pd.DataFrame(MinMaxScaler().fit_transform(ohe_ac_data), columns = all_ac_data.columns)

    all_ac_data_mms.describe().T.tail(n=3); # 유닛테스트(비활성) - 스케일링 후 결과

    orange_X = all_ac_data_mms.copy()

    orange_y = orange_X.pop('churned')

    #print(orange_X.columns) # 유닛테스트(비활성) - X에 y 포함 여부

    

    ## 2. 고객 이탈 예측 - 최종 학습된 우리 모델 활용

    orange_prediction = final_model.predict(orange_X)

    orange_predict_proba = final_model.predict_proba(orange_X)

    

    ## 3. 이탈 예상 고객 목록

    data_final = data.copy()

    data_final.insert(loc=0, column='churn_probability', value=orange_predict_proba[:,1]*100)

    data_final = data_final.sort_values(by=['churn_probability'], ascending=False)

    churn_list = data_final[:int(sum(orange_prediction))]

    no_churn_list = data_final[int(sum(orange_prediction)):]

    

    return orange_prediction, churn_list, no_churn_list
# 여기서 시프트+엔터!

orange_prediction, churn_list, no_churn_list = data_to_predict(data_raw)
print('오렌지 담당자님!', len(data_raw), '분의 고객에 대한 이탈 예측을 원하셨죠?')

print('안타깝게도 이 중 이탈이 예상되는 분은', int(sum(orange_prediction)), '분이었어요. (저는 정말 0이 나오길 바랐어요)')

print('전체 중', sum(orange_prediction)/len(data_raw)*100.0, "%에 가까운 비율이에요. 과연 어떤 분들이 오렌지텔레콤과의 이별을 준비하시는지 아래에서 상세히 살펴볼게요.")
churn_list.head(n=3).T
no_churn_list.tail(n=3).T
# 1. 이탈/비이탈 예상 고객 총이용금액 확인

charge_list = []

for i in churn_list.columns:

    if 'charge' in i:

        charge_list.append(i)

total_charge_no_churn = no_churn_list[charge_list].sum(axis=1)

total_charge_churn = churn_list[charge_list].sum(axis=1)



# 2. 히스토그램

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5))



plt.subplot(1, 2, 1)

plt.xlabel('Total Charge ($)')

plt.ylabel('Frequency')

plt.title('Frequency by Total Charge')

plt.hist(total_charge_no_churn, color='darkcyan', alpha=0.31)

plt.hist(total_charge_churn, color='darkcyan')

plt.title('No Churn vs. Churn')



plt.subplot(1, 2, 2)

plt.xlabel('Total Charge ($)')

plt.ylabel('Frequency')

plt.title('Frequency by Total Charge')

plt.hist(total_charge_churn, color='darkcyan')

plt.title('Churn Only')



fig.tight_layout()
# 히스토그램

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5))



plt.subplot(1, 2, 1)

plt.xlabel('Account Length (month)')

plt.ylabel('Frequency')

plt.title('Frequency by Account Length')

plt.hist(no_churn_list.account_length, color='darkcyan', alpha=0.31)

plt.hist(churn_list.account_length, color='darkcyan')

plt.title('No Churn vs. Churn')



plt.subplot(1, 2, 2)

plt.xlabel('Account Length (month)')

plt.ylabel('Frequency')

plt.title('Frequency by Account Length')

plt.hist(churn_list.account_length, color='darkcyan')

plt.title('Churn Only')



fig.tight_layout()
# 총이용금액, 계약 유지기간, 이탈 가능성을 고려한 대응시급성 산출기

def probability_to_emergency(churn_list=churn_list, charge_weight=0.7, churn_weight=0.85):

    churn_list_new = churn_list.copy() # 반복실험용 원본데이터 저장

    # 0. 총이용금액 열 추가

    charge_list = []

    for i in churn_list_new.columns:

        if 'charge' in i:

            charge_list.append(i)

    churn_list_new.insert(loc=1, column='total_charge', value=churn_list_new[charge_list].sum(axis=1))



    # 1. 고객가치 계산에 필요한 데이터 구성

    churn_list_new.churn_probability = churn_list_new.churn_probability/100.0

    emer_list = pd.DataFrame(MinMaxScaler().fit_transform(churn_list_new.iloc[:,1:4].drop('state', axis=1)),

                             columns=['total_charge', 'account_length'], index=churn_list_new.index)

    emer_list = emer_list.join(churn_list_new.churn_probability)



    # 2. 총이용금액 / 계약 유지기간 열에 가중치 적용하여 고객가치 계산 후 이탈 예상 리스트에 추가

    emer_list['customer_value'] = emer_list.total_charge*(charge_weight) + emer_list.account_length*(1-charge_weight)

    churn_list_new.insert(loc=0, column='customer_value', value=emer_list.customer_value)



    # 3. 이탈 가능성 / 고객가치 열에 가중치 적용하여 대응시급성 계산 후 이탈 예상 리스트에 추가

    emer_list['emergency'] = emer_list.churn_probability*churn_weight + emer_list.customer_value*(1-churn_weight)

    churn_list_new.insert(loc=0, column='emergency', value=emer_list.emergency)



    # 4. 대응시급성 지표에 따라 전체 열 재배치 / 소수점 수치 백분율로 바꾸기

    churn_list_new = churn_list_new.sort_values(by=['emergency'], ascending=False)

    churn_list_new.iloc[:,:3] = churn_list_new.iloc[:,:3]*100

    churn_list_new = churn_list_new.round(2)

    

    return churn_list_new
new_churn_list = probability_to_emergency(churn_list=churn_list, charge_weight=0.5, churn_weight=0.85)
new_churn_list.drop('state', axis=1).head(5).T.head(5)
# 이탈 예상 고객 순위 변동 비율

num_change = sum(churn_list.index != new_churn_list.index)

ratio_change = num_change/len(churn_list)*100

ratio_change = np.around(ratio_change, 2)



print("이탈가능성(churn_probability)에서 대응시급성(emergency)을 기준으로 많은 분들의 순위가 바뀌었어요.")

print("정확하게는", num_change, "분이고요. 총", len(churn_list), "분의 이별 예상 고객 중", ratio_change, "%에 가까워요.")

print("즉 이런 과정을 거치지 않은 채 이별 가능성만을 기준으로 고객분들께 접근했다면,")

print("많은 시간과 노력을 잘못된 순서에 따라 썼을 수 있다는 의미랍니다.")
# 잠시 쉬는 줄 (여긴 읽지 마시고 눈을 쉬게 해주세요!)
# 두 문서(인텔을 이기다, 배민라이더스 앱 리뉴얼 효과 분석 제안서)와 관련된 질문이 있다면 댓글로 알려주세요

## 그동안 kaggle 속에 꼭꼭 숨어 지냈는데 이번 기회에 처음으로 이렇게 공개활동을 하네요

### 워낙 고수분들이 많은 프로의 세계라 겁나지만 설레기도 합니다