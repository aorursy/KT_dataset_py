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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix,classification_report,roc_curve,accuracy_score,auc

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from keras.utils import np_utils

from keras.models import Sequential, load_model

from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten

df = pd.read_csv("../input/pima-diabetes-dataset/Diabetes.csv")
print(df.columns.tolist())
title_mapping = {'YES':1,'NO':0}

df[' Class variable']=df[' Class variable'].map(title_mapping)
#0처리



z=(df == 0).sum(axis=0)

z=pd.DataFrame(z)

z.columns=['Zeros Count']

z.drop(' Class variable',inplace=True)

z.plot(kind='bar',stacked=True, figsize=(10,5),grid=True)
col=['n_pregnant','glucose_conc','bp','skin_len','insulin','bmi','pedigree_fun','age','Output']

df.columns=col

df.head()
#0 제거 후 nan 으로 변환

col=['glucose_conc','bp','insulin','bmi','skin_len']

for i in col:

    df[i].replace(0, np.nan, inplace= True)
#결과확인

df.isnull().sum()
def median_target(var):   

    temp = df[df[var].notnull()]

    temp = temp[[var, 'Output']].groupby(['Output'])[[var]].median().reset_index()

    return temp
median_target('insulin')
median_target('glucose_conc')
median_target('skin_len')
median_target('bp')
median_target('bmi')
#중간값으로 NaN 채워넣기

df.loc[(df['Output'] == 0 ) & (df['insulin'].isnull()), 'insulin'] = 102.5

df.loc[(df['Output'] == 1 ) & (df['insulin'].isnull()), 'insulin'] = 169.5

df.loc[(df['Output'] == 0 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 107

df.loc[(df['Output'] == 1 ) & (df['glucose_conc'].isnull()), 'glucose_conc'] = 140

df.loc[(df['Output'] == 0 ) & (df['skin_len'].isnull()), 'skin_len'] = 27

df.loc[(df['Output'] == 1 ) & (df['skin_len'].isnull()), 'skin_len'] = 32

df.loc[(df['Output'] == 0 ) & (df['bp'].isnull()), 'bp'] = 70

df.loc[(df['Output'] == 1 ) & (df['bp'].isnull()), 'bp'] = 74.5

df.loc[(df['Output'] == 0 ) & (df['bmi'].isnull()), 'bmi'] = 30.1

df.loc[(df['Output'] == 1 ) & (df['bmi'].isnull()), 'bmi'] = 34.3
#아웃라이어를 찾기 위한 박스플롯

plt.style.use('ggplot') # Using ggplot2 style visuals 



f, ax = plt.subplots(figsize=(11, 15))



ax.set_facecolor('#fafafa')

ax.set(xlim=(-.05, 200))

plt.ylabel('Variables')

plt.title("Overview Data Set")

ax = sns.boxplot(data = df, 

  orient = 'h', 

  palette = 'Set2')
#중간값을 활용한 아웃라이어 제거

sns.boxplot(df.n_pregnant)



df['n_pregnant'].value_counts()
median_target('n_pregnant')
#인위조정 : 임신횟수

df.loc[(df['Output'] == 0 ) & (df['n_pregnant']>13), 'n_pregnant'] = 2

df.loc[(df['Output'] == 1 ) & (df['n_pregnant']>13), 'n_pregnant'] = 4
#확인사살

df['n_pregnant'].value_counts()
#혈압. 이하 모든 변수에 대해서 반복하기 때문에 큰 설명은 하지 않겠습니다.

sns.boxplot(df.bp)
median_target('bp')
df.loc[(df['Output'] == 0 ) & (df['bp']<40), 'bp'] = 70

df.loc[(df['Output'] == 1 ) & (df['bp']<40), 'bp'] = 74.5
df.loc[(df['Output'] == 0 ) & (df['bp']>103), 'bp'] = 70

df.loc[(df['Output'] == 1 ) & (df['bp']>103), 'bp'] = 74.5
sns.boxplot(df.bp)
sns.boxplot(df.skin_len)
median_target('skin_len')
df.loc[(df['Output'] == 0 ) & (df['skin_len']>38), 'skin_len'] = 27

df.loc[(df['Output'] == 1 ) & (df['skin_len']>38), 'skin_len'] = 32

df.loc[(df['Output'] == 0 ) & (df['skin_len']<20), 'skin_len'] = 27

df.loc[(df['Output'] == 1 ) & (df['skin_len']<20), 'skin_len'] = 32
sns.boxplot(df.skin_len)
sns.boxplot(df.bmi)
median_target('bmi')
df.loc[(df['Output'] == 0 ) & (df['bmi']>48), 'bmi'] = 30.1

df.loc[(df['Output'] == 1 ) & (df['bmi']>48), 'bmi'] = 34.3
sns.boxplot(df.bmi)
sns.boxplot(df.pedigree_fun)
median_target('pedigree_fun')
df.loc[(df['Output'] == 0 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.336

df.loc[(df['Output'] == 1 ) & (df['pedigree_fun']>1), 'pedigree_fun'] = 0.449

sns.boxplot(df.pedigree_fun)
median_target('age')
df.loc[(df['Output'] == 0 ) & (df['age']>61), 'age'] = 27

df.loc[(df['Output'] == 1 ) & (df['age']>61), 'age'] = 36
VarCorr = df.corr()

print(VarCorr)

sns.heatmap(VarCorr,xticklabels=VarCorr.columns,yticklabels=VarCorr.columns)
#테스트셋, 트레인셋 분리

X = df.drop(['n_pregnant','skin_len','insulin','pedigree_fun','Output'], 1)

y = df['Output']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X.head(5)
std = StandardScaler()

x_train = std.fit_transform(x_train)

x_test = std.transform(x_test)
model=SVC(kernel='rbf')

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
model=SVC(kernel='linear')

model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))
regressor=LogisticRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
clf = KNeighborsClassifier(n_neighbors=3) 

clf.fit(x_train,y_train)  

print(clf.score(x_test,y_test))
print(classification_report(y_test,y_pred))
classifier=RandomForestClassifier()

classifier.fit(x_train,y_train)
Y_pred=classifier.predict(x_test)

confusion_matrix(y_test,Y_pred)
accuracy_score(y_test,Y_pred)
model = Sequential()

model.add(Dense(32,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(64,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(64,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(128,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(128,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(256,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(256,input_shape=(x_train.shape[1],)))

model.add(Activation('relu'))

model.add(Dense(2))

model.add(Activation('softmax'))
print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',

              optimizer="sgd",metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=10, epochs=50, verbose=1, validation_data=(x_test, y_test))

loss, accuracy = model.evaluate(x_test,y_test, verbose=0)

print("Loss : "+str(loss))

print("Accuracy :"+str(accuracy*100.0))