import pandas as pd

import numpy as np

import seaborn as sns

## 주피터 노트북에 그래프를 그리기 위해 추가해줘야하는 magic cell 

## matplotlib = 파이썬에서 데이터 차트를 그리기 위한 라이브러리

import matplotlib.pyplot as plt

%matplotlib inline



## 데이터를 읽어옴

train_origin=pd.read_csv("../input/train.csv")

test_origin=pd.read_csv("../input/test.csv")

sample_sub=pd.read_csv("../input/gender_submission.csv")



train = train_origin.copy()

test = test_origin.copy()
##데이터 정보 파악

train.info()
## Name, Sex, Ticket, Cabin, Embarked 는 object 타입. 확인 필요

## 데이터를 보면서 형상을 좀 파악

train.head()
## Sex 나 Embarked 는 카테고라이징이 되어 있는 것 같으니 패스

## 이름은 일단... Mr, Mrs, Miss 요정도로만 카테고라이징 해서 경향 파악을 해보자

## Ticket 은 딱봐도 답이 없으니 제거

## Cabin 은 AXX BXX 이런 식인것 같은니 앞자리만 때어서 경향파악을 해보자



train['name_title'] = train.Name.str.extract(' ([A-Za-z]+)\.')

train['cabin_prefix'] = train.Cabin.str.extract('(^[A-Z])')

train.drop(['Ticket'],axis=1,inplace=True)



train.head()

## 대강 분리 어려운 데이터에 대한 변겨은 했으니 어떻게 분류가 될수 있을지 확인해보자

train['name_title'].value_counts()
train['cabin_prefix'].value_counts()
train['Sex'].value_counts()
train['Embarked'].value_counts()
## 음 어느정도 카테고라이징 할 수 있게 된것 같군.

## 하지만 count 가 1 이고 이런것들이 존재하니 한번 더 묶어줘야 하나 싶은데 일단 그래프로 찍어보자



train_name_title = train.pivot_table(index="name_title", columns="Survived", aggfunc="size")



sns.heatmap(train_name_title, cmap=sns.light_palette(

    "gray", as_cmap=True), annot=True)

## 찍어 놓고 보니 유의미한 차이가 있는 name_title 이 있고 아닌것들이 있다.

## Miss, Mr, Mrs, Rev 빼고 나머지는 다 뭉쳐서 Other 로 바꾸자

## 추후에 test 에서 다른 title 이 발견되어 나중에 함수로 통합할때는 replace 가 아닌 list 에 있는것 빼고 다 바꾸는 코드로 수정



train['name_title'].replace(['Master','Dr','Mlle','Col','Major','Jonkheer','Mme','Sir','Don','Capt','Countess','Lady','Ms'],'Other', inplace=True)

train['name_title'].value_counts()

## 그다음 카테고라이징이 애매했던 cabin 에 대해서도 heatmap 으로 확인해보자

## NaN 도 포함되어 있었으니 NaN 은 Z 로 치환해서 확인해보자



train['cabin_prefix'].fillna('Z', inplace=True)



train_cabin = train.pivot_table(index="cabin_prefix", columns="Survived", aggfunc="size")



sns.heatmap(train_cabin, cmap=sns.light_palette(

    "gray", as_cmap=True), annot=True)

## T 는 숫자가 너무 적으니 Z 에 편입 시키자

train['cabin_prefix'].replace('T','Z', inplace=True)

## 카테고라이징이 힘든 데이터들을 모두 변환 완료 하였다.

## 이제 남아있는 NaN 을 처리하기위해 info 를 확인해보자

train.info()
## Age 와 Embarked 에 NaN 이 있다.

## Age 에 mean 을 넣을 것인지 median 을 넣을지 살펴보기 위해 Age 그래프를 확인해보자



train['Age'].plot.kde()



## Age 는 분포를 보니 mean 으로 채워도 될것 같아 mean 으로 채우기로 함

## Embarked 는 가장 많은 카운트의 값으로 채우기로 함



train['Age'].fillna((train['Age'].mean()), inplace=True)

train['Embarked'].fillna((train['Embarked'].value_counts().idxmax()), inplace=True)

## test set 에 있는 값들 중 위에서 처리 못한 NaN 값이 있는 컬럼이 있는지 확인

test.info()
## Fare 에 NaN 이 존재함으로 어떻게 채울지 판단하기 위해 그래프 확인

## 학습은 train 기반으로 하기 때문에 train 데이터로 확인해야함

train['Fare'].plot.kde()

## 그래프를 보니 돈을 엄청 낸 사람들도 있고 해서 평균은 애매하고

## 가장 많은 사람들이 낸 요금이 더 의미있을 것 같아 그렇게 하기로 함.

## 일단 train 기준으로 코드르 짜놓고 나중에 test 에 train 에서 뽑은 값을 넣어주는 함수를 만들것

train['Fare'].fillna((train['Fare'].value_counts().idxmax()), inplace=True)

## 카테고리화 되어 있는 데이터를 숫자로 변경

## 각각 카테고리가 수적인 연속성이 있는것이 아니기 때문에 One-hot encoding 을 하기로 함

embarked_1hot = pd.get_dummies(train["Embarked"], prefix="Embarked")

name_title_1hot = pd.get_dummies(train["name_title"], prefix="Name")

cabin_1hot = pd.get_dummies(train["cabin_prefix"], prefix="Cabin")

sex_1hot = pd.get_dummies(train["Sex"], prefix="Sex")





train = pd.concat([train, embarked_1hot], axis=1)

train = pd.concat([train, name_title_1hot], axis=1)

train = pd.concat([train, cabin_1hot], axis=1)

train = pd.concat([train, sex_1hot], axis=1)



## 필요없는 컬럼을 다 드롭 시키자

## 어차피 여자아니면 남자니까 Sex_male 도 제거하자

train.drop(['Name'],axis=1,inplace=True)

train.drop(['Embarked'],axis=1,inplace=True)

train.drop(['Sex'],axis=1,inplace=True)

train.drop(['Cabin'],axis=1,inplace=True)

train.drop(['name_title'],axis=1,inplace=True)

train.drop(['cabin_prefix'],axis=1,inplace=True)

train.drop(['Sex_male'],axis=1,inplace=True)



train.head()

## 위 작업을 추후 반복을 해야하니 클래스로 만들어 두자

class Cleaner:

    def __init__(self):

        self.fill_age = 0

        self.fill_embarked = ''

        self.fill_fare = 0



    def prepare_data(self, input_data, istrain=True):

        data = input_data.copy()

        data['name_title'] = data.Name.str.extract(' ([A-Za-z]+)\.')

        data['cabin_prefix'] = data.Cabin.str.extract('(^[A-Z])')

        data.drop(['Ticket'],axis=1,inplace=True)

        

        ## 중간에 test 에서 다른 성들이 발견되어 이것 이외에는 다 변경으로 코드 수정

        ## 결과값이 낮아서 Rev 도 other 로 편입. 데이터 양이 적었기 때문에

        data.loc[~data["name_title"].isin(['Miss','Mr', 'Mrs']), "name_title"] = "Other"

        data['cabin_prefix'].fillna('Z', inplace=True)

        data['cabin_prefix'].replace('T','Z', inplace=True)

        

        if istrain:        

            self.fill_age = data['Age'].mean()

            self.fill_embarked = data['Embarked'].value_counts().idxmax()

            self.fill_fare = data['Fare'].value_counts().idxmax()

            

        data['Age'].fillna(self.fill_age, inplace=True)

        data['Embarked'].fillna(self.fill_embarked, inplace=True)

        data['Fare'].fillna(self.fill_fare, inplace=True)

        

        data_embarked_1hot = pd.get_dummies(data["Embarked"], prefix="Embarked")

        data_name_title_1hot = pd.get_dummies(data["name_title"], prefix="Name")

        data_sex_1hot = pd.get_dummies(data["Sex"], prefix="Sex")

        ## 정확도가 생각보다 낮아서 cabin 값 아예 누락 시킴 ( NaN 이 애초에 너무 많은 데이터 였기 때문에 )

#         data_cabin_1hot = pd.get_dummies(data["cabin_prefix"], prefix="Cabin")

        

        data = pd.concat([data, data_embarked_1hot], axis=1)

        data = pd.concat([data, data_name_title_1hot], axis=1)

        data = pd.concat([data, data_sex_1hot], axis=1)

#         data = pd.concat([data, data_cabin_1hot], axis=1)

        

        data.drop(['Name'],axis=1,inplace=True)

        data.drop(['PassengerId'],axis=1,inplace=True)

        data.drop(['Embarked'],axis=1,inplace=True)

        data.drop(['Sex'],axis=1,inplace=True)

        data.drop(['Cabin'],axis=1,inplace=True)

        data.drop(['name_title'],axis=1,inplace=True)

        data.drop(['cabin_prefix'],axis=1,inplace=True)

        data.drop(['Sex_male'],axis=1,inplace=True)

    

        return data
## 함수를 이용해서 데이터를 다시 전처리 해보자.

cleaner = Cleaner()

train_prepared = cleaner.prepare_data(train_origin, istrain=True)

test_prepared = cleaner.prepare_data(test_origin, istrain=False)



train_prepared.head()
import tensorflow as tf

from tensorflow import keras

from sklearn.preprocessing import StandardScaler



## 텐서플로우를 이용하여 학습을 시켜보자

## train data 에서 y 값 분리

train_prepared_y = train_prepared['Survived']

train_prepared.drop(['Survived'],axis=1,inplace=True)

test_prepared_y = sample_sub['Survived']

## feature scaling 를 해준다. logistic regression 인지라 정규화로 해주기로 함

sc = StandardScaler()

train_prepared = sc.fit_transform(train_prepared)

test_prepared = sc.fit_transform(test_prepared)



train_prepared[:5]
## 모델을 구성한다. 아직 모델에 대한것은 잘 알지 못해서 적절하게만 구성해본다.



model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(16, activation = 'relu', input_dim=13),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(8, activation = 'relu'),

    tf.keras.layers.Dense(1, activation = 'sigmoid')



])



model.summary()
## 90% 넘으면 트레이닝을 종료한다.

class EpochEndCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('acc')>0.90):

            print("\nReached 90% accuracy so cancelling training!")

            self.model.stop_training = True
## 모델 fit 수행



endCallback = EpochEndCallback()

model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(train_prepared, train_prepared_y, batch_size = 10, nb_epoch = 20, verbose=1, validation_split = 0.2, callbacks=[endCallback])

## 실제 테스트 셋으로 평가를 해본다.

test_loss, test_accuracy = model.evaluate(test_prepared, test_prepared_y)

print('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))
## 캐글 제출용으로 파일을 만든다.

predict = model.predict(test_prepared)

predict_survived = (predict > 0.6).astype(int).reshape(test.shape[0])
## f1 score 를 측정한다.

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score



print('f1 score : ',f1_score(test_prepared_y.values, predict_survived))

print('precision : ',precision_score(test_prepared_y.values, predict_survived))

print('recall : ',recall_score(test_prepared_y.values, predict_survived))

print('auc : ',roc_auc_score(test_prepared_y.values, predict_survived))
## recall, percision  그래프를 그리기 위해 준비 한다.



predict_survived_confusion = predict * 100

predict_survived_confusion = predict_survived_confusion.astype('uint8')

predict_survived_confusion[:5]
from sklearn.metrics import precision_recall_curve



precision, recalls, thresholds = precision_recall_curve(test_prepared_y.values, predict_survived_confusion)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="recall", linewidth=2)

    plt.xlabel("thresholds", fontsize=16)

    plt.legend(loc="upper left", fontsize=16)

    plt.ylim([0, 1])



def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("recall", fontsize=16)

    plt.ylabel("precision", fontsize=16)

    plt.axis([0, 1, 0, 1])
plt.figure(figsize=(8, 4))

plot_precision_recall_vs_threshold(precision, recalls, thresholds)

plt.xlim([0, 100])

plt.show()
plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precision, recalls)

plt.show()
predict_survived_real = (predict > 0.45).astype(int).reshape(test.shape[0])



sample_sub['Survived'] = predict_survived_real

sample_sub.to_csv("submit.csv", index=False)

sample_sub.head()