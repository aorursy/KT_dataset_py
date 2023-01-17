# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

print("NumPy version: {}". format(np.__version__))

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print("pandas version: {}". format(pd.__version__))

import seaborn as sns



print("seaborn version: {}". format(sns.__version__))





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')  # training dataframe

test = pd.read_csv('../input/test.csv')  # testing dataframe

train.head()
#문자열로 구성되어있는 열과, 보면 count가 다른 Age열처럼 결측값 존재하는 열도 있음.

train.describe()
#test 데이터 는 대략적으로 418개의 데이터가 존재 하며, 결측값 또한 보인다.

test.describe()
#Age, Cabin, Embarked 각각 null값이 존재.

train.isnull().sum()
test.isnull().sum()


# visualization

import seaborn as sns

import matplotlib.pyplot as plt



#subplot으로 그래프를 1행 2열로 2개 생성한다.

f, ax=plt.subplots(1, 2, figsize=(18,8))



#생존한 사람과 사망한 사람의 비율을 원그래프로 보여준다.

train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Survived')

ax[0].set_ylabel('')



#생존한 사람과 사망한 사람을 count하여 막대그래프로 보여준다.

sns.countplot('Survived',data=train, ax=ax[1])

ax[1].set_title('Survived')

plt.show()



#약 61.6%가 사망하였고, 약 38.4%가 생존하였다.
train['Age'].hist(bins=20, figsize= (15, 6), grid=False)
plt.figure(figsize=(10, 10))

sns.heatmap(train.corr(), linewidths=0.01, square= True,

           annot = True, cmap= plt.cm.viridis, linecolor = 'white')

plt.title('Correlation between Features')

plt.show()
#모델링한 데이터들을 Feature Engineering을 하게 되면, test데이터도 똑같이 수정해야하기 때문에 편리성을 위해 실행시키겠습니다.

#원본의 데이터를 건드는 것이기 때문에, train_test_data를 수정하면 원본의 데이터돋도 같이 수정됩니다.

train_test_data = [train, test]
f,ax=plt.subplots(1,3,figsize=(18,5))

#각 성별에 따라서 Survived 비율을 그래프로 나타낸다.

train['Survived'][train['Pclass']==1].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

train['Survived'][train['Pclass']==2].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)

train['Survived'][train['Pclass']==3].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[2],shadow=True)

ax[0].set_title('Survived (Pclass : 1)')

ax[1].set_title('Survived (Pclass : 2)')

ax[2].set_title('Survived (Pclass : 3)')

plt.show()



#Pclass가 1인 경우는 63%가 살았으나, 2와 3은 각각 52.7%, 75.8%가 죽었다.
train.groupby('Pclass').mean()
f,ax=plt.subplots(1,2,figsize=(18,8))

#각 성별에 따라서 Survived 비율을 그래프로 나타낸다.

train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')

plt.show()



#male의 Survived 비율은 약 81.1%가 사망하였고, 약 18.9%가 생존하였다.

#female의 Survived 비율은 약 25.8%가 사망하였고, 약 74.2%가 생존하였다. 



#남자는 약 81%가 사망하였고, 여자는 26%가 사망하였다.

#남자가 더 많이 사망한 것으로 보아 Lady - first가 시행되었을 것이다.
from sklearn.preprocessing import LabelBinarizer

#먼저 양적변수로 바꿔 줄 함수 호출

encoder=LabelBinarizer()

#변환된 열을 생성하여 적용

train['Sex']=encoder.fit_transform(train['Sex'])

test['Sex']=encoder.fit_transform(test['Sex'])

train.head(10)

#이렇게 하면 male 즉, 남자가 1 여자가 0 이 된다.
#이름 전체를 해석하기 난해하기 때문에 사람에게 붙는 호칭으로 분석을 하고자 함.

for dataset in train_test_data:

    dataset['Name'] = dataset['Name'].str.extract('([A-Za-z]+)\.', expand=True)
titles=train['Name'].unique()

titles
titles_test=test['Name'].unique()

titles_test
#Mr부터 Rev까지는 어느 정도의 인원이 있으므로 다른 인원들은 비슷한 호칭으로 통일 시켜준다.

train['Name'].value_counts()
replace_name = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',

          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
train.replace({'Name' : replace_name}, inplace=True)
test.replace({'Name' : replace_name}, inplace=True)
#각 Name에 따라 생존확률에 따라 생존확률이 높을 수록 높은 점수를 매긴다.

train[['Name', 'Survived']].groupby(['Name'], as_index=False).mean()
#사용 시 'Rev'인 경우 인식을 하지 못함. 꼭 ""(큰따음표) 로 감싸줄 것

score_name = {"Rev" : 0, "Mr" : 1, "Dr" : 2, "Master" : 3, "Miss" : 4, "Mrs" : 5}

for dataset in train_test_data:

    dataset['Name'] = dataset['Name'].map(score_name)

train.head(10)
#먼저 이름에 따라서 나이에 Null 값을 채워 준다.

titles = [0,1,2,3,4,5]

for title in titles:

    age_to_impute = train.groupby('Name')['Age'].median()[titles.index(title)]

    train.loc[(train['Age'].isnull()) & (train['Name'] == title), 'Age'] = age_to_impute

train['Age'].isnull().sum()
for title in titles:

    age_to_impute = train.groupby('Name')['Age'].median()[titles.index(title)]

    test.loc[(test['Age'].isnull()) & (test['Name'] == title), 'Age'] = age_to_impute

test['Age'].isnull().sum()
#나이와 생존에 관한 histogram



f, ax = plt.subplots(1, 1, figsize=(12, 5))

sns.kdeplot(train[train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(train[train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()



#나이에 따라서 20~30대 일수록 많이 생존하기도 했으며 또한 많이 죽기도 했다.

#나이가 어릴수록 생존확률이 높다.
#나이 구간을 나누어 준다.

for dataset in train_test_data:

    dataset['Age_bin'] = pd.cut(train['Age'], 5)
train[['Age_bin','Survived']].groupby(['Age_bin'], as_index=False).mean().sort_values(by='Survived')
group_names = [4,1,3,2,0]

#Age를 score로 환산한다.

for dataset in train_test_data:

    dataset['Age'] = pd.cut(train['Age'], 5, labels=group_names)
#많이 살았을 수록 점수가 더 높은 것을 확인 할 수 있다.

train[['Age','Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived')
#Embarked같은 경우 2명의 null값이 있는데 이는 S에서 탑승한 인원이 제일 많기 때문에 S로 채워넣겠습니다.

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')
#많이 살았을 수록 점수가 더 높은 것을 확인 할 수 있다.

train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived')
#이것도 생존율이 높을 수록 높은 점수를 할당 한다.

embarked_mapping = {"S": 0, "Q": 1, "C": 2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

test['FamilySize'] = test['SibSp'] + test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived')
replace_family = {8 : 0, 11 : 0, 6:1, 5: 2, 1:3, 7:4, 2 : 5, 3 :6, 4:7}
for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].apply(lambda x: replace_family.get(x))
test['Fare'].fillna(train['Fare'].median(), inplace = True)
train
for dataset in train_test_data:

    dataset['Fare_bin'] = pd.cut(train['Fare'], 4)
train[['Fare_bin','Survived']].groupby(['Fare_bin'], as_index=False).mean().sort_values(by='Survived')
group_fare = [0,2,1,3]

for dataset in train_test_data:

    dataset['Fare'] = pd.cut(train['Fare'], 4, labels=group_fare)
train[['Fare','Survived']].groupby(['Fare'], as_index=False).mean()
train['Cabin'].value_counts().sort_index()
#Cabin의 데이터를 사용하기위해 제일 앞 알파벳만 이용을 함.

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].str[:1]
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()

Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()

Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()

df=pd.DataFrame([Pclass1, Pclass2, Pclass3])

df.index = ['1st class', '2nd class', '3rd class']

df.plot(kind='bar', stacked=True, figsize=(10,5))
# fill missing Fare with median fare for each Pclass

train.loc[(train['Cabin'].isnull())&(train['Pclass']==1), 'Cabin'] = 'E'

train.loc[(train['Cabin'].isnull())&(train['Pclass']==2), 'Cabin'] = 'D'

train.loc[(train['Cabin'].isnull())&(train['Pclass']==3), 'Cabin'] = 'F'

test.loc[(test['Cabin'].isnull())&(test['Pclass']==1), 'Cabin'] = 'E'

test.loc[(test['Cabin'].isnull())&(test['Pclass']==2), 'Cabin'] = 'D'

test.loc[(test['Cabin'].isnull())&(test['Pclass']==3), 'Cabin'] = 'F'
Survived_1 = train[train['Survived']==1]['Cabin'].value_counts()

Survived_0 = train[train['Survived']==0]['Cabin'].value_counts()

df=pd.DataFrame([Survived_1, Survived_0])

df.index = ['1', '0']

df.plot(kind='bar', stacked=True, figsize=(10,5))
cabin_mapping = {"B": 0,  "C": 1 , "A": 2, "T": 3, "E": 4, "D": 5, "F": 6, "G": 7}

for dataset in train_test_data:

    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
train.head(10)
features_drop = ['Ticket', 'SibSp', 'Parch', 'Age_bin', 'Fare_bin']

train = train.drop(features_drop, axis=1)

test = test.drop(features_drop, axis=1)
train.tail(10)
test.head(10)
train.to_csv('train.csv', index=False)

test.to_csv('test.csv', index=False)
train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)

train['Fare'] = train['Fare'].astype(int)

test['Fare'] = test['Fare'].astype(int)
train.info()
test.info()
# Importing Classifier Modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



import numpy as np
import tensorflow as tf

import keras

from keras.models import Sequential

from keras.layers.core import Dense

np.random.seed(42)



print('tensorflow version : ', tf.__version__)

print('keras version : ', keras.__version__)
#모델링의 간편화를 위해..

train = train.drop(['PassengerId'], axis=1)
train.info()
x_data = train.values[:, 1:]

y_data = train.values[:, 0]



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.3, random_state = 42)
model = Sequential()

model.add(Dense(255, input_shape=(8, ), activation = 'relu'))

model.add(Dense((1), activation = 'sigmoid'))

model.compile(loss='mse', optimizer='Adam', metrics = ['accuracy'])

model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))
hist = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
plt.figure(figsize=(12,8))

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.legend(['loss','val_loss', 'acc','val_acc'])

plt.show()
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
train_data = train.drop('Survived', axis=1)

target = train['Survived']

train_data.shape, target.shape
clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# kNN Score

round(np.mean(score)*100, 2)
clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# decision tree Score

round(np.mean(score)*100, 2)
clf = RandomForestClassifier(n_estimators=13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Random Forest Score

round(np.mean(score)*100, 2)
clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
# Naive Bayes Score

round(np.mean(score)*100, 2)
#SVC는 C와 gamma에 따라 조금 씩 달라지는데 이때 C=10일때 accuracy가 더 높게 나옴.

clf = SVC(C=10)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(score)
round(np.mean(score)*100,2)
model.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()

prediction = model.predict(test_data)
prediction = pd.Series(prediction.reshape(418, )).map(lambda x: 1 if x >= 0.5 else 0)
prediction
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": prediction

    })



submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head(10)