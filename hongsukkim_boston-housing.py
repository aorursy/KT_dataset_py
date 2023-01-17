import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston



boston = load_boston()

print(boston.DESCR) #데이터 수 : 506 , 속성 수 : 13
dfX = pd.DataFrame(boston.data, columns=boston.feature_names)

dfy = pd.DataFrame(boston.target, columns=["MEDV"])

df = pd.concat([dfX, dfy], axis=1)

print(df.shape)
df = df.sample(frac=1)



# 데이터 분할 : 506개 중 400개는 train 데이터, 106개는 test 데이터로 사용.

train = df[:401]

test = df[401:]

train.head()
train.info()
fig, axs = plt.subplots(4,4, figsize=(16, 10))

fig.subplots_adjust(hspace = 0.5, wspace=.2)

axs = axs.ravel()

i=0

for col in train.columns:

    axs[i].plot(train[col])

    axs[i].set_title(col)

    i = i + 1
# 범주형 데이터는 one hot encoding으로 변환

print(train[['CHAS','MEDV']].groupby('CHAS', as_index=False).mean())

print(test[['CHAS','MEDV']].groupby('CHAS', as_index=False).mean())

train = pd.get_dummies(train, columns=['CHAS'], prefix='CHAS')

test = pd.get_dummies(test, columns=['CHAS'], prefix='CHAS')

#test['CHAS_1.0']=0 # test의 CHAS 컬럼에 1 인 값이 없어서 컬럼 추가함 ㅠㅠ

train.head()
# 나머지 데이터는 정규화

cols = ['CRIM','ZN','INDUS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']

train_temp = train[cols]

test_temp = test[cols]
# standardization 평균=0, 표준편차=1으로 만듬

# z = ( X - mean) / std

mean = train_temp.mean(axis=0)

std = train_temp.std(axis=0)

train_temp = (train_temp - mean) / std

test_temp = (test_temp - mean) / std

train_temp.head()
# 데이터 확인

fig, axs = plt.subplots(4,3, figsize=(16, 10))

fig.subplots_adjust(hspace = 0.5, wspace=.2)

axs = axs.ravel()

i=0

for col in train_temp.columns:

    axs[i].plot(train_temp[col])

    axs[i].set_title(col)

    i = i + 1
plt.plot(train['CRIM'],label='before scaling')

plt.plot(train_temp['CRIM'],label='after scaling')

plt.legend()
# apply

for i, feature in enumerate(train_temp.columns):

    train[feature] = train_temp[feature]

    test[feature] = test_temp[feature]

train.head()
from keras import models, layers

from keras.models import Sequential

from keras.layers.core import Dense

from keras.optimizers import Adam
# 모델 구성 함수

def build_model():

    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(14,))) # 14 : medv를 제외한 모든 컬럼 수

    model.add(Dense(64, activation='relu'))

    model.add(Dense(1)) # no activation : 0-1 사이의 값을 예측하는 것이 아니라 모든 범위에서 실제 값을 예측하도록 함

    # mse : mean squared error 평균 제곱 오차. 예측과 타깃 사이 제곱. 회귀 문제에서 많이 사용.

    # mae : mean absolute error 평균 절대 오차. 예측과 타깃 사이 절대값.

    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 

    return model
train_targets = train['MEDV']

train_data = train.drop(['MEDV'], axis=1)

test_targets = test['MEDV']

test_data = test.drop(['MEDV'], axis=1)

print(train_data.shape)

print(test_data.shape)
# 1. hold-out

# 데이터 분할 1-299, 300-400

train_data1= train_data[:300]

train_targets1= train_targets[:300]

val_data1= train_data[300:]

val_targets1= train_targets[300:]

print(train_data1.shape)

print(val_data1.shape)



# 모델 구성

model1 = build_model()

# 훈련

history1 = model1.fit(train_data1, train_targets1,

                      validation_data=(val_data1, val_targets1),

                      epochs=100, batch_size=10, verbose=0)
plt.plot(history1.history['mean_absolute_error'],label='mae')

plt.plot(history1.history['val_mean_absolute_error'],label='val_mae')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend()
# 2. k-fold

k = 4

num_val_samples = len(train_data) // k

all_mae_histories =[]

all_val_mae_histories=[]

for i in range(k):

    print('fold #',i)

    val_data2 = train_data[i * num_val_samples : (i + 1) * num_val_samples]

    val_targets2 = train_targets[i * num_val_samples : (i + 1) * num_val_samples]

    train_data2 = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]], axis=0)

    train_targets2 = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]], axis=0)

    model2 = build_model()

    # 훈련  

    history2 = model2.fit(train_data2, train_targets2, 

                          validation_data = (val_data2, val_targets2), 

                          epochs = 100, batch_size = 10, verbose=0) # verbose=0 훈련 과정 출력 안함    

    mae_history = history2.history['mean_absolute_error']

    val_mae_history = history2.history['val_mean_absolute_error']

    all_mae_histories.append(mae_history)

    all_val_mae_histories.append(val_mae_history)
avg_mae_history =[np.mean([x[i] for x in all_mae_histories]) for i in range(100)]

avg_val_mae_history =[np.mean([x[i] for x in all_val_mae_histories]) for i in range(100)]

plt.plot(range(1,len(avg_mae_history)+1), avg_mae_history,label='model2_mae')

plt.plot(range(1,len(avg_val_mae_history)+1), avg_val_mae_history,label='model2_val_mae')

plt.plot(history1.history['mean_absolute_error'],'y-',label='model1_mae')

plt.plot(history1.history['val_mean_absolute_error'],'y-',label='model1_val_mae')

plt.legend()

plt.show()
plt.plot(history1.history['mean_absolute_error'],'y-',label='model1_mae')

plt.plot(history1.history['val_mean_absolute_error'],'y-',label='model1_val_mae')

plt.plot(all_mae_histories[0],'g-',label='model2_mae0')

plt.plot(all_mae_histories[1],'g-',label='model2_mae1')

plt.plot(all_mae_histories[2],'g-',label='model2_mae2')

plt.plot(all_mae_histories[3],'g-',label='model2_mae3')

plt.plot(all_val_mae_histories[0],'g-',label='model2_val_mae0')

plt.plot(all_val_mae_histories[1],'g-',label='model2_val_mae1')

plt.plot(all_val_mae_histories[2],'g-',label='model2_val_mae2')

plt.plot(all_val_mae_histories[3],'g-',label='model2_val_mae3')

plt.legend(bbox_to_anchor=(1.5,1))

plt.show()
predict1 = model1.predict(test_data)

predict2 = model2.predict(test_data)

#predict3 = model3.predict(test_data)

plt.plot(test_targets.values, 'co', label='target')

plt.plot(predict1, label='1')

plt.plot(predict2, label='2')

#plt.plot(predict3, label='3')

plt.legend(bbox_to_anchor=(1,1))
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(test_targets.values, predict1))

print(mean_absolute_error(test_targets.values, predict2))

#print(mean_absolute_error(test_targets.values, predict3))
# KFold 종류 비교

from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit



x_data = np.array([[1,2],[2,4],[3,2],[4,4],[5,4],[6,2],[7,4],[8,4],[9,2],[10,4],[11,2],[12,4],[13,2],[14,4],[15,4]

                  ,[16,2],[17,4],[18,4],[19,2],[20,4]])

y_data = np.array([1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5])

kfold1 = KFold(n_splits=4)

kfold2 = KFold(n_splits=4, shuffle=True, random_state=42)

skfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

shufflesplit = StratifiedShuffleSplit(n_splits=4, random_state=42, test_size=5)



print("KFold")

for train_index, test_index in kfold1.split(x_data, y_data):

    print("TRAIN:", train_index, "TEST:", test_index)

    

print("KFold shuffle")

for train_index, test_index in kfold2.split(x_data, y_data):

    print("TRAIN:", train_index, "TEST:", test_index)

    

print("sKFold")

for train_index, test_index in skfold.split(x_data, y_data):

    print("TRAIN:", train_index, "TEST:", test_index)



print("Shuffle Split")

for train_index, test_index in shufflesplit.split(x_data, y_data):

    print("TRAIN:", train_index, "TEST:", test_index)

print(x_data.shape)

print(y_data.shape)
# 3. k-fold suffle

all_mae_histories2 =[]

all_val_mae_histories2=[]

cv = KFold(n_splits=4, shuffle=True,random_state=42)

for i , (idx_train, idx_test) in enumerate(cv.split(train_data, train_targets)):

    print('fold #',i)    

    train_data3 = train_data.iloc[idx_train]

    train_targets3 = train_targets.iloc[idx_train]

    val_data3 = train_data.iloc[idx_test]

    val_targets3 = train_targets[idx_test]

    model3 = build_model()

    # 훈련  

    history3 = model3.fit(train_data3, train_targets3, 

                          validation_data = (val_data3, val_targets3), 

                          epochs = 100, batch_size = 10, verbose=0) # verbose=0 훈련 과정 출력 안함    

    mae_history = history3.history['mean_absolute_error']

    val_mae_history = history3.history['val_mean_absolute_error']

    all_mae_histories2.append(mae_history)

    all_val_mae_histories2.append(val_mae_history)
avg_mae_history2 =[np.mean([x[i] for x in all_mae_histories2]) for i in range(100)]

avg_val_mae_history2 =[np.mean([x[i] for x in all_val_mae_histories2]) for i in range(100)]

plt.plot(range(1,len(avg_mae_history2)+1), avg_mae_history2,label='mae')

plt.plot(range(1,len(avg_val_mae_history2)+1), avg_val_mae_history2,label='val_mae')

plt.legend()

plt.show()
plt.plot(history1.history['mean_absolute_error'],'y-',label='model1_mae')

plt.plot(history1.history['val_mean_absolute_error'],'y-',label='model1_val_mae')

plt.plot(all_mae_histories[0],'g-',label='model2_mae0')

plt.plot(all_mae_histories[1],'g-',label='model2_mae1')

plt.plot(all_mae_histories[2],'g-',label='model2_mae2')

plt.plot(all_mae_histories[3],'g-',label='model2_mae3')

plt.plot(all_val_mae_histories[0],'g-',label='model2_val_mae0')

plt.plot(all_val_mae_histories[1],'g-',label='model2_val_mae1')

plt.plot(all_val_mae_histories[2],'g-',label='model2_val_mae2')

plt.plot(all_val_mae_histories[3],'g-',label='model2_val_mae3')

plt.plot(all_mae_histories2[0],'b-',label='model3_mae0')

plt.plot(all_mae_histories2[1],'b-',label='model3_mae1')

plt.plot(all_mae_histories2[2],'b-',label='model3_mae2')

plt.plot(all_mae_histories2[3],'b-',label='model3_mae3')

plt.plot(all_val_mae_histories2[0],'b-',label='model3_val_mae0')

plt.plot(all_val_mae_histories2[1],'b-',label='model3_val_mae1')

plt.plot(all_val_mae_histories2[2],'b-',label='model3_val_mae2')

plt.plot(all_val_mae_histories2[3],'b-',label='model3_val_mae3')

plt.legend(bbox_to_anchor=(1,1))

plt.show()
predict1 = model1.predict(test_data)

predict2 = model2.predict(test_data)

predict3 = model3.predict(test_data)

plt.plot(test_targets.values, 'co', label='target')

plt.plot(predict1, label='1')

plt.plot(predict2, label='2')

plt.plot(predict3, label='3')

plt.legend(bbox_to_anchor=(1,1))
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(test_targets.values, predict1))

print(mean_absolute_error(test_targets.values, predict2))

print(mean_absolute_error(test_targets.values, predict3))