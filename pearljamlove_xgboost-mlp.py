import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D # 원래 CNN할려구 했는데 귀찮아서 안했음
from keras import optimizers

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import re
dataTrain = pd.read_csv('../input/train.csv')
dataTest = pd.read_csv('../input/test.csv')
### 학습, 테스트자료에 Null 값을 확인
for col in dataTrain.columns.tolist():
    length = len(dataTrain[col])
    lengthNull = len(dataTrain[col].dropna())
    print('Null values of {:s} is {:d}'.format(col, length - lengthNull))
 # 학습자료에는 Age, Cabin 그리고 Embarked 에 Null 값이 있다.

for col in dataTest.columns.tolist():
    length = len(dataTest[col])
    lengthNull = len(dataTest[col].dropna())
    print('Null values of {:s} is {:d}'.format(col, length - lengthNull))
 # Age, Fare have null values on test data.
 # 테스트자료에는 Age, Fare 그리고 Cabin에 Null 값이 있다.
### 이름(Name)에 호칭을 찾기
nameWords = []
for i in dataTrain['Name']:
    i = re.sub('[,.()"]', '', i).strip().split(' ')
    for j in i:
        nameWords.append(j)
nameTitles = Series(nameWords).value_counts()[:10]
print(nameTitles) # 대충 다음과 같은 호칭을 찾을 수 있다 - Mr, Miss, Mrs, Master
del i, j, nameWords, nameTitles
### 호칭(Mr, Miss, Mrs, Master)을 숫자형태로 변환
def nameClassifier(dataset):
    nameList = []
    for name in dataset:
        if re.search('Master', name) != None:
            nameList.append(0)
        elif re.search('Mrs', name) != None:
            nameList.append(1)
        elif re.search('Mr', name) != None:
            nameList.append(2)
        elif re.search('Miss', name) != None:
            nameList.append(3)
        else:
            nameList.append(4)
    return Series(nameList)
dataTrain['segName'] = nameClassifier(dataTrain['Name'])
### 전체 가족숫자를 만들어 봅시다 !
def familySize(dataset):
    return dataset['SibSp'] + dataset['Parch']
dataTrain['Family'] = familySize(dataTrain)
### 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'segName', 'Family' 에 대한 생존/사망
fig, ax = plt.subplots(2, 4, figsize=(18, 8))
colnames= ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'segName', 'Family']
i = 0
for row in range(2):
    for col in range(4):
        try:
            temp = dataTrain.groupby([colnames[i], 'Survived']).agg({'PassengerId':'count'}).reset_index()
            sns.barplot(data=temp, x=colnames[i], y='PassengerId', hue='Survived', ax=ax[row, col])
            i += 1
        except:
            continue
plt.show()
### 'SibSp', 'Parch', 'Family', 'Embarked'별, Pclass 등급확인
fig, ax = plt.subplots(2, 2, figsize=(18, 8))
colnames= ['SibSp', 'Parch', 'Family', 'Embarked']
i = 0
for row in range(2):
    for col in range(2):
        try:
            # temp = dataTrain.groupby(['Pclass', 'Survived', colnames[i]]).agg({'PassengerId':'count'}).reset_index()
            temp = dataTrain.groupby(['Pclass', colnames[i]]).agg({'PassengerId':'count'}).reset_index()
            # temp['Class_Survived'] = temp['Pclass'].astype(str) + '_' + temp['Survived'].astype(str)
            sns.barplot(data=temp, x=colnames[i], y='PassengerId', hue='Pclass', ax=ax[row, col])
            i += 1
        except:
            continue
plt.show()
### 남성/여성(Sex)를 0/1로 변환
def sexClassifier(dataset):
    return Series([1 if i == 'male' else 0 for i in dataset])
dataTrain['segSex'] = sexClassifier(dataTrain['Sex'])
### (Pclass)를 0-2로 변환
def pclassClassifier(dataset):
    return dataset -1
dataTrain['segPclass'] = pclassClassifier(dataTrain['Pclass'])
### SibSp를 숫자형태로 변환
print(dataTrain.pivot_table(index=['SibSp'], columns=['Survived'], values=['PassengerId'], aggfunc='count', fill_value=0))
 # 생존자가 더 많은 분류에 1을, 사망자가 더 많은 경우 0을
def sibClassifier(dataset):
    return Series([1 if (1 <= i) & (i <= 2) else 0 for i in dataset])
    # return Series([0 if 0 == i else 1 for i in dataset])
dataTrain['segSib'] = sibClassifier(dataTrain['SibSp'])
### Parch를 숫자형태로 변환
print(dataTrain.pivot_table(index=['Parch'], columns=['Survived'], values=['PassengerId'], aggfunc='count', fill_value=0))
 # 생존자가 더 많은 분류에 1을, 사망자가 더 많은 경우 0을
def parClassifier(dataset):
    return Series([0 if (1 <= i) & (i <= 3) else 1 for i in dataset])
    # return Series([0 if 0 == i else 1 for i in dataset])
dataTrain['segPar'] = parClassifier(dataTrain['Parch'])
### Family 를 0/1 숫자형태로 변환
def familyClassifier(dataset):
    return Series([0 if (1 <= i) & (i <= 3) else 1 for i in dataset])
dataTrain['segFamily'] = familyClassifier(dataTrain['Family'])
### 객실(Cabin) 유무로 생존/사망 확인 : 객실이 없는 경우 사망율이 높다
print(dataTrain[dataTrain['Cabin'].isnull()].groupby(['Pclass', 'Survived'])['PassengerId'].count(), '\n')
print(dataTrain[dataTrain['Cabin'].notnull()].groupby(['Pclass', 'Survived'])['PassengerId'].count())
def cabinClassifier(dataset):
    temp = pd.DataFrame(dataset)
    temp.loc[temp[temp.columns[0]].notnull(), temp.columns] = 1
    temp.loc[temp[temp.columns[0]].isnull(), temp.columns] = 0
    return temp[temp.columns]
dataTrain['segCabin'] = cabinClassifier(dataTrain['Cabin'])
### 요금을 quartile을 기준으로 숫자로 변환 : 플롯은 그리기 귀찮아서 그런데... 대충 비쌀 수록 안죽는다.
def fareClassifier(dataset):
    return pd.qcut(dataset, [0, .25, .5, .75, 1], labels=[0, 1, 2, 3]).astype(int)
dataTrain['segFare'] = fareClassifier(dataTrain['Fare'])
### 나이(Age)의 Null 값을 채우기 (Null이 아닌 레코드의 4가지 Feature 조합으로 나이의 중위수을 채운다)
dataAgeSet = dataTrain.groupby(['Pclass', 'segName', 'segSib', 'segPar']).agg({'Age':'median'}).reset_index() # 조합별 나이의 중위수를 구함 
dataTrain = dataTrain.merge(dataAgeSet, on=['Pclass', 'segName', 'segSib', 'segPar'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTrain['Age'] = dataTrain['Age'].where(dataTrain['Age'].notnull(), dataTrain['Age_y']) # Null 값을 채움
del dataTrain['Age_y']
### 나이별 히스토그램인데, 그냥 그려봄
fig, ax = plt.subplots(1,2, figsize=(12, 6))
ax[0].hist(dataTrain['Age'])
ax[1].boxplot(dataTrain['Age'])
plt.show()
# 나이를 quartile을 기준으로 숫자로 변환
def ageClassifier(dataset):
    return pd.qcut(dataset, [0, .25, .5, .75, 1], labels=[0, 1, 2, 3]).astype(int)
dataTrain['segAge'] = ageClassifier(dataTrain['Age'])
### 탑승항구(Embarked)의 Null 값을 채우기 (Null이 아닌 레코드의 4가지 Feature 조합으로 항구의 최빈값을 채운다)
dataEmbarkedSet = dataTrain.groupby(['Pclass', 'segName', 'segSib', 'segPar']).agg({'Embarked':lambda x: x.mode()}).reset_index()  # 조합별 항구의 최빈값을 구함 
dataTrain = dataTrain.merge(dataEmbarkedSet, on=['Pclass', 'segName', 'segSib', 'segPar'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTrain['Embarked'] = dataTrain['Embarked'].where(dataTrain['Embarked'].notnull(), dataTrain['Embarked_y']) # Null 값을 채움
del dataTrain['Embarked_y']

def ebkClassifier(dataset): # 탑승항구(Embarked)를 숫자로 변환
    lists = []
    for row in dataset:
        if re.search('S', row) != None:
            lists.append(0)
        elif re.search('C', row) != None:
            lists.append(1)
        elif re.search('Q', row) != None:
            lists.append(2)
        else:
            lists.append(3)
    return Series(lists)
dataTrain['segEmbarked'] = ebkClassifier(dataTrain['Embarked'])
### 분석에 필요한 Feature만 추출
dataTrainTemp = dataTrain[['segName', 'segSex', 'segPclass', 'segSib', 'segFamily', 'segPar', 'segFare', 'segAge']] # 몇번 돌리다보니 이 조합이 가장 좋음

def transformEncoding(dataset): # One hot encoidng 형태로 바꿔주는 함수
    temp1 = pd.DataFrame()
    for col in dataset.columns:
        if len(dataset[col].unique()) == 2:
            temp0 = dataset[col]
        else:
            temp0 = pd.get_dummies(dataset[col])
            temp0.columns = [col + str(i) for i in temp0.columns]
        temp1 = pd.concat([temp1, temp0], axis=1)
    return temp1

X = transformEncoding(dataTrainTemp).values # 학습 Features
y = dataTrain['Survived'].values.reshape(len(dataTrainTemp), 1).astype(int) # 답안 Label
### 모형함수 : 원래 여러개 만들어야 하는데, 귀찮아서 2개로 때웁니다. 회사일 때문에 잠을 못자서 피곤해요... (CNN은 만들다가 망해서 빼버림)
def modelXGB(X_train, y_train):
    model = xgb.XGBClassifier(max_depth=50, n_estimators=700, n_jobs=-1, learning_rate=0.005, gamma=0.01) # 이게 잴 좋더라구요.
    model.fit(X_train, y_train)
    return model

def modelMLP(X_train, y_train, n_epoch, n_batch, optm): # 그냥 특징없는 인공신경망
    model = Sequential()
    model.add(Dense(256, input_shape=(X_train.shape[1], ), activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optm, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=n_epoch, batch_size=n_batch)
    return model
### 파라메터는 이정도로~
n_epoch = 100
n_batch = 50
optm = optimizers.Adam(lr=0.0001)
### K-Fold Cross Validation : 피곤해도 교차검증은 해야죠.
kfold = StratifiedKFold(n_splits=5, random_state=0)
accuracyXGB, accuracyMLP, accuracyENS = [], [], []
for train, test in kfold.split(X, y, groups=y):
    
    # XGBoost
    model1 = modelXGB(X[train], y[train])
    accuracy1 = model1.score(X[test], y[test])
    print(accuracy1)
    accuracyXGB.append(accuracy1)
    
    # MLP
    model2 = modelMLP(X[train], y[train], n_epoch, n_batch, optm)
    accuracy2 = model2.evaluate(X[test], y[test])[1]
    print(accuracy2)
    accuracyMLP.append(accuracy2)
    
    # 2개 합친거
    pred1 = model1.predict_proba(X[test])[:, 1]; pred1 = pred1.reshape(len(pred1))
    pred2 = model2.predict_proba(X[test]); pred2 = pred2.reshape(len(pred2))
    pred = (pred1 + pred2) / 2
    pred = np.array([1 if i > 0.5 else 0 for i in pred])
    accuracy3= accuracy_score(pred, y[test].reshape(len(y[test])))
    accuracyENS.append(accuracy3)
### 3개 모형 (CNN / MLP / 두개 소프트 배깅) : 이젠 XGBoost 뿐이야 !
accTable1 = DataFrame({'accuracy':accuracyXGB, 'model':'XGB'})
accTable2 = DataFrame({'accuracy':accuracyMLP, 'model':'MLP'})
accTable3 = DataFrame({'accuracy':accuracyENS, 'model':'Ensemble'})
accTable = pd.concat([accTable1, accTable2, accTable3], axis=0)
del accTable1, accTable2, accTable3
accTableAgg = accTable.groupby(['model']).agg({'accuracy':['mean', 'max', 'min', 'std']})
print(accTableAgg)
### 3개 모형 (CNN / MLP / 두개 소프트 배깅) 시각화로 봅시다 : 이젠 XGBoost 뿐이야 !
fig, ax = plt.subplots(1,3, figsize=(16,5))
plt.suptitle('K-Fold Cross Validation (K=5)')
for i, col in enumerate(accTable['model'].unique().tolist()):
    temp = accTable[accTable['model'] == col]
    ax[i].bar(range(len(temp)), temp['accuracy'])

    ax[i].set_ylim(accTable['accuracy'].min()*.95, accTable['accuracy'].max())

    ax[i].set_title('Accuracy : {}'.format(col))
plt.show()
### 테스트 자료 전처리 : 위에서 만든 함수들로 전처리함
dataTest['segName'] = nameClassifier(dataTest['Name'])
dataTest['Family'] = familySize(dataTest)
dataTest['segSex'] = sexClassifier(dataTest['Sex'])
dataTest['segPclass'] = pclassClassifier(dataTest['Pclass'])
dataTest['segSib'] = sibClassifier(dataTest['SibSp'])
dataTest['segPar'] = parClassifier(dataTest['Parch'])
dataTest['segFamily'] = familyClassifier(dataTest['Family'])
dataTest['segCabin'] = cabinClassifier(dataTest['Cabin'])
### 테스트 자료에 나이(Age) Null 값을 체움
dataAgeSet = dataTrain.groupby(['Pclass', 'segName', 'segSib', 'segPar']).agg({'Age':'median'}).reset_index() # 조합별 나이의 중위수를 구함 
dataTest = dataTest.merge(dataAgeSet, on=['Pclass', 'segName', 'segSib', 'segPar'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTest['Age'] = dataTest['Age'].where(dataTest['Age'].notnull(), dataTest['Age_y']) # Null 값을 채움
del dataTest['Age_y']

dataAgeSet = dataTrain.groupby(['Pclass', 'segName', 'segSib']).agg({'Age':'median'}).reset_index() # 조합별 나이의 중위수를 구함 
dataTest = dataTest.merge(dataAgeSet, on=['Pclass', 'segName', 'segSib'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTest['Age'] = dataTest['Age'].where(dataTest['Age'].notnull(), dataTest['Age_y']) # Null 값을 채움
del dataTest['Age_y']

dataAgeSet = dataTrain.groupby(['segName', 'segSib', 'segPar']).agg({'Age':'median'}).reset_index() # 조합별 나이의 중위수를 구함 
dataTest = dataTest.merge(dataAgeSet, on=['segName', 'segSib', 'segPar'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTest['Age'] = dataTest['Age'].where(dataTest['Age'].notnull(), dataTest['Age_y']) # Null 값을 채움
del dataTest['Age_y']
### 테스트 자료에 요금(Fare) Null 값을 체움
dataFareSet = dataTrain.groupby(['Pclass', 'segName', 'segSib', 'segPar']).agg({'Fare':'median'}).reset_index() # 조합별 나이의 중위수를 구함 
dataTest = dataTest.merge(dataFareSet, on=['Pclass', 'segName', 'segSib', 'segPar'], how='left', suffixes=('','_y')) # 상기구한 조합을 Join
dataTest['Fare'] = dataTest['Fare'].where(dataTest['Fare'].notnull(), dataTest['Fare_y']) # Null 값을 채움
del dataTest['Fare_y']
### 나머지 Feature 변환
dataTest['segFare'] = fareClassifier(dataTest['Fare'])
dataTest['segAge'] = ageClassifier(dataTest['Age'])
dataTest['segEmbarked'] = ebkClassifier(dataTest['Embarked'])
### One Hot Encoding으로 변환
dataTestTemp = dataTest[['segName', 'segSex', 'segPclass', 'segSib', 'segFamily', 'segPar', 'segFare', 'segAge']]
X_test = transformEncoding(dataTestTemp).values
### 모형 : XGB로 갑시다
model1 = modelXGB(X, y)
pred = model1.predict(X_test)
pred = Series(pred)
submission = pd.read_csv('../input/sample_submission.csv')
submission['Survived'] = pred
submission.head() # 올~~~
# 두근두근 제출완료
submission.to_csv('./submission.csv', index=False)

