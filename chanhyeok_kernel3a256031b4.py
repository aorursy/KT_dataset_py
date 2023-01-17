import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=2.5) # 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.

import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') # 워닝 메세지를 생략해 줍니다. 차후 버전관리를 위해 필요한 정보라고 생각하시면 주석처리 하시면 됩니다.



# 이미지, 소리 등의 Rich Output을 브라우저에서 볼 수 있게 해줌

%matplotlib inline 
os.listdir("../input")



df_train = pd.read_csv('../input/train.csv') # GTD 포함 12 Cols

df_test = pd.read_csv('../input/test.csv')   # 11 COls

df_submit = pd.read_csv('../input/sample_submission.csv')



df_train.shape, df_test.shape, df_submit.shape
df_train.head()

df_train.describe()

df_test.head()

df_test.describe()
df_submit.head()

# df_submit.describe()
df_train.dtypes
f, ax = plt.subplots(1, 2, figsize=(18, 8))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0])

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, 

               size=6, aspect=1.5)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(19, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
# Age distribution withing classes

plt.figure(figsize=(8, 6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age Distribution within classes')

plt.legend(['1st Class', '2nd Class', '3rd Class'])
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

    

plt.figure(figsize=(18, 10))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.00)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1 # 자신을 포함해야하니 1을 더합니다
print("Maximum size of Family: ", df_train['FamilySize'].max())

print("Minimum size of Family: ", df_train['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)



sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
df_train["Age"].isnull().sum()
df_train['Initial']= df_train.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations

df_test['Initial']= df_test.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r') #Checking the Initials with the
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
df_train.groupby('Initial')['Survived'].mean().plot.bar()
# 평균값 대입



df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mr'),'Age'] = 33

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Mrs'),'Age'] = 36

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Master'),'Age'] = 5

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Miss'),'Age'] = 22

df_train.loc[(df_train.Age.isnull())&(df_train.Initial=='Other'),'Age'] = 46



df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mr'),'Age'] = 33

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Mrs'),'Age'] = 36

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Master'),'Age'] = 5

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Miss'),'Age'] = 22

df_test.loc[(df_test.Age.isnull())&(df_test.Initial=='Other'),'Age'] = 46
df_train.isnull().sum()[df_train.isnull().sum() > 0]
df_train['Embarked'].fillna('S', inplace=True) # Embarked를 S로 치환
df_train.isnull().sum()[df_train.isnull().sum() > 0]
df_test.isnull().sum()[df_test.isnull().sum() > 0]
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean() # test-set의 fare nan value를 평균값으로 치환합니다.
df_test.isnull().sum()[df_test.isnull().sum() > 0]
df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})



df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})





df_train['Embarked'].isnull().any() , df_train['Embarked'].dtypes



df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})



heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})



del heatmap_data


df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train.head()
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.head()
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
#importing all the required ML packages

from sklearn.ensemble import RandomForestClassifier # 유명한 randomforestclassfier 입니다. 

from sklearn import metrics # 모델의 평가를 위해서 씁니다

from sklearn.model_selection import train_test_split # traning set을 쉽게 나눠주는 함수입니다.
X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_train.shape, X_test.shape
X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2019)
from keras.models import Sequential

from keras.layers import BatchNormalization

from keras.layers.core import Dense, Dropout

from keras.optimizers import Adam, SGD
nn_model = Sequential()

nn_model.add(Dense(32,activation='relu',input_shape=(13,)))

nn_model.add(Dropout(0.3))

nn_model.add(BatchNormalization())

nn_model.add(Dense(64,activation='relu'))

nn_model.add(Dropout(0.3))

nn_model.add(BatchNormalization())

nn_model.add(Dense(64,activation='relu'))

nn_model.add(Dropout(0.2))

nn_model.add(BatchNormalization())

nn_model.add(Dense(1,activation='sigmoid'))



Loss = 'binary_crossentropy'

nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])

nn_model.summary()
history = nn_model.fit(X_tr,y_tr,

                    batch_size=64,

                    epochs=500,

                    validation_data=(X_vld, y_vld),

                    verbose=1)
hists = [history]

hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)

hist_df.index = np.arange(1, len(hist_df)+1)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))

axs[0].plot(hist_df.val_acc, lw=5, label='Validation Accuracy')

axs[0].plot(hist_df.acc, lw=5, label='Training Accuracy')

axs[0].set_ylabel('Accuracy')

axs[0].set_xlabel('Epoch')

axs[0].grid()

axs[0].legend(loc=0)

axs[1].plot(hist_df.val_loss, lw=5, label='Validation MLogLoss')

axs[1].plot(hist_df.loss, lw=5, label='Training MLogLoss')

axs[1].set_ylabel('MLogLoss')

axs[1].set_xlabel('Epoch')

axs[1].grid()

axs[1].legend(loc=0)

fig.savefig('hist.png', dpi=300)

plt.show();
submission = pd.read_csv('../input/sample_submission.csv')

submission.head()
prediction = nn_model.predict(X_test)

prediction = prediction > 0.5

prediction = prediction.astype(np.int)

prediction = prediction.T[0]

prediction.shape
submission['Survived'] = prediction

submission.to_csv('my_nn_submission.csv', index=False)
os.listdir("../input")