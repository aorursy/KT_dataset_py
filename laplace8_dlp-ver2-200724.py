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
import matplotlib.pyplot as plt #시각화 도구 module

pd.set_option('max_columns',150,'max_rows',20) # 결과값을 몇줄까지 출력할지 설정



# #검은 배경에서 그래프 요소들을 밝은 색으로 변환 (dark mode에서만 사용)

# params = {"text.color" : "g",

#           "ytick.color" : "w",

#           "xtick.color" : "w",

#           "axes.titlecolor" : "w",

#           "axes.labelcolor" : "w",

#           "axes.edgecolor" : "w"}

# plt.rcParams.update(params)

# train = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/train.csv') # train data 호출

# test = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/test.csv') # test data 호출

# submission = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/submission.csv') # submission data 호출



# train.head() # train data 형태 위에서 5줄까지 조회, 데이터 전처리를 시작하기 전 각 column description 확인



train = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/train.csv')

test = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/test.csv')

submission = pd.read_csv('/kaggle/input/dlp-private-competition-2nd-ai-study/submission.csv')



train.head()
test.head() #test set 형태 확인









# train.shape, test.shape, submission.shape # 3가지 set의 크기(행,열 개수) 확인





train.shape, test.shape, submission.shape
print(round(train.isnull().sum() / len(train) *100,2).sort_values(ascending=False)[:10])

print(round(test.isnull().sum() / len(train) *100,2).sort_values(ascending=False)[:10])
# isnull().sum(): 각 columns의 빈 값의 개수 확인

# len(train): train 의 행 개수

# * 100: 비율을 백분율로 바꿔줌

# round(train, n) data를 소숫점 n번째 자리까지 표기

# sort_values(ascending=False): data 내림차순 정렬

# [:10] 가장 큰 10개 data 표시

print(round(train.isnull().sum() / len(train) *100,2).sort_values(ascending=False)[:10])

print(round(test.isnull().sum() / len(train) *100,2).sort_values(ascending=False)[:10])





# null_col: train data에서 빈값이 가장 많은 10개 column 지정

# for(반복문)문으로 train, test set의 빈 data를 각 column의 최빈값(mode)으로 채워줌

# [0]: 최빈값 중 가장 앞에 나온 값 사용



null_col = ['C78','C50','C18','C82','C38','C5','C88','C76','C110','C10']

for each in null_col:

    train[each] = train[each].fillna(train[each].mode()[0])

    test[each] = test[each].fillna(test[each].mode()[0])





# # column중 index와 END_TM은 반도체 두께에 영향을 미치기 힘들기 때문에 삭제



# train = train.drop(['index', 'END_TM'],1)

# test = test.drop(['index', 'END_TM'],1)



# train.head() # Column이 잘 지워졌나 확인



train = train.drop(['index', 'END_TM'],1)

test = test.drop(['index', 'END_TM'],1)



train.head()



# # data.unique(): 각 column에 무슨 값들이 있는지(목록) list로 확인



# print(train['A'].unique())

# print(test['A'].unique())



print(train['A'].unique())

print(test['A'].unique())
# # set 형태로 test set A열에는 있지만 train set A열에는 없는 요소를 선별

# # 학습도 안 시키고 예측을 하라고 할 수는 없음



# only_test = set(test['A']).difference(set(train['A']))

# only_test



only_test = set(test['A']).difference(set(train['A']))

only_test
# # 학습에 부적합하므로 A열 삭제



# train = train.drop(['A'],1)

# test = test.drop(['A'],1)



train = train.drop(['A'],1)

test = test.drop(['A'],1)



train.head()

# # B열도 똑같이 .unique()로 목록 조회



print(train['B'].unique())

print(test['B'].unique())





# # one hot encoding을 통해 문자 데이터를 숫자 데이터로 변환 (컴퓨터가 읽을 수 있게)



# train_B = pd.get_dummies(train['B'], prefix='B') # 목록 수 만큼 columns 생성

# train = pd.concat([train, train_B], axis=1) # 숫자로 나타낸 column들을 train data에 열 방향 붙임

# train = train.drop(['B','B_F','B_J','B_H','B_K','B_G','B_D','B_C','B_E'],1)

# # 숫자로 바꿔줬으니 B열 삭제, 모델에 넣기 위해 train, test의 column 개수가 같아야 해서

# # one hot encoding 후 train에 남는 B_F~B_E열 삭제

# train.head() # 변환됐는지 확인

train_B = pd.get_dummies(train['B'], prefix='B')

train = pd.concat([train, train_B], axis=1)

train = train.drop(['B','B_F','B_J','B_H','B_K','B_G','B_D','B_C','B_E'],1)

train.head()
# test_B = pd.get_dummies(test['B'], prefix='B') # test에도 one hot encoding 적용

# test = pd.concat([test, test_B], axis=1)

# test = test.drop(['B'],1)



# test.head()



test_B = pd.get_dummies(test['B'], prefix='B')

test = pd.concat([test, test_B], axis=1)

test = test.drop(['B'],1)

test.head()

# # [Feature Importance Finding Process - It takes 5~20min to run]

# features = pd.DataFrame(0., columns = ['MSE_train','MSE_valid','diff_train','diff_valid','diff_sum' ],

#                            index = ['C'+str(x) for x in range(1,118)])



# from sklearn.model_selection import train_test_split

# import lightgbm as lgb

# from sklearn.metrics import mean_squared_error

# lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144, 

#                           learning_rate=0.005,n_estimators=720, max_depth=13,

#                           metric='rmse', is_training_metric=True, max_bin=55,

#                           bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)



# for each in range(1,118):

#     train_tmp = train.drop('C'+str(each),1)

    

#     y = train_tmp['Y']

#     X = train_tmp.drop(['Y'],1)

    

#     X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)



#     lgbm.fit(X_train, y_train)

    

#     print(str(each))

#     pred_train = lgbm.predict(X_train)

#     train_mse = mean_squared_error(pred_train, y_train)

#     print(train_mse)

    

#     pred_valid = lgbm.predict(X_valid)

#     valid_mse = mean_squared_error(pred_valid, y_valid)

#     print(valid_mse)

    

#     ['MSE_train', 'MSE_valid', 'diff_train', 'diff_valid', 'diff_sum']

#     features['MSE_train']['C'+str(each)] = train_mse

#     features['MSE_valid']['C'+str(each)] = valid_mse  

#     features['diff_train']['C' + str(each)] = 8.576 - train_mse

#     features['diff_valid']['C' + str(each)] = 16.414 - valid_mse

#     features['diff_sum']['C' + str(each)] = (8.576 - train_mse)+(16.414 - valid_mse)



# features.to_csv("result_feature_importance.csv", index=False)    

# plt.plot(features)

# plt.show()
# # feature importance 확인 결과, C59열은 예측의 성능을 떨어뜨리므로 열 삭제

# train = train.drop(['C59'],1)

# test = test.drop(['C59'],1)
# train['Y'].describe() # 예측할 Y값의 평균, 표준편차, 최댓값, 4분위값, 최솟값 확인

train['Y'].describe()



fig, ax = plt.subplots() # 예측할 Y값의 분포 시각화

ax.scatter(x=train.index, y=train['Y'])

plt.ylabel('Thickness', fontsize=13)

plt.show()





#  # 두께값이 0인 반도체는 세상에 존재하지 않아서 Y=0인 행 삭제

# print(train.shape)

# count = 0

# for x in range(0,11618):

#     if train['Y'][x] == 0:

#         train = train.drop(x)

#         count +=1



# train.shape, count # Y=0인 행 삭제 전후 shape 비교, 삭제된 행의 개수 count



print(train.shape)

count = 0

for x in range(0,11618):

    if train['Y'][x] == 0:

        train = train.drop(x)

        count +=1

train.shape, count        



from sklearn.model_selection import train_test_split # train set 을 4개로 분할해주는 모듈

import lightgbm as lgb # 요즘 선호하는 가볍고 성능좋은 모델

from sklearn.metrics import mean_squared_error # 평가척도인 MSE 모듈



# y = train['Y']  # 결과값

# X = train.drop(['Y'],1) # 결과값을 제외한 예측에 사용되는 자료



y = train['Y']

X = train.drop(['Y'],1)



#훈련재료,검증재료, 훈련 답, 검증 답 으로 4분할 (보통 훈련용 70%, 검증용 30%)

X_train, X_valid, y_train, y_valid = train_test_split (X,y, random_state=0)



#사용할 모델(Light GBM) 정의

lgbm = lgb.LGBMRegressor (objective = 'regression', num_leaves=144,

                             learning_rate=0.005,n_estimators=720, max_depth=13,

                             metric='rmse', is_training_metric=True, max_bin=55,

                             bagging_fraction=0.8, verbose=-1, bagging_freq=5, feature_fraction=0.9)



# # 모델로 예측(fitting)

# lgbm.fit(X_train, y_train)

lgbm.fit(X_train, y_train)



# #훈련 결과값과 실제값의 편차 제곱 평균으로 평가

pred_train = lgbm.predict(X_train)

train_mse = mean_squared_error(pred_train, y_train)

print('Train_mse: ', train_mse)



# #검증 결과값과 실제값의 편차 제곱 평균으로 평가

pred_valid = lgbm.predict(X_valid)

valid_mse = mean_squared_error(pred_valid, y_valid)

print('Valid_mse: ', valid_mse)
# #test 데이터를 모델에 적용해서 예측

pred_test = lgbm.predict(test)

submission.head() # submission 의 형태 확인





pred_test

submission_final = pd.concat([submission, pred_test], axis=1)
# submission의 Y값이 다 0이므로 Y열 먼저 제거

submission = submission.drop('Y',1)

pred_test = pd.DataFrame(pred_test)  # DataFrame 으로 바꿔주기 전엔 pred_test 가 array 형태



# test set의 예측값을 submission Y열 뺀 자리에 병합, axis=1 : 열 방향을 의미

submission_final = pd.concat([submission, pred_test], axis=1)

# 병합한 예측값 column 이름을 Y로 지정

submission_final.columns = ['index','Y']

# 제출할 csv 형태의 파일 생성

submission_final.to_csv('submission_final.csv', index=False)

# 결과파일이 알맞은 형태로 나왔는지 확인

submission_final.head()






