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
import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import keras

import sklearn



plt.style.use('seaborn')

sns.set(font_scale=1)



import missingno as msno



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
os.listdir('../input/2019-1st-ml-month-with-kakr')
df_train = pd.read_csv('../input/2019-1st-ml-month-with-kakr/train.csv')

df_test = pd.read_csv('../input/2019-1st-ml-month-with-kakr/test.csv')

df_submit = pd.read_csv('../input/2019-1st-ml-month-with-kakr/sample_submission.csv')



df_train.shape, df_test.shape, df_submit.shape
df_train.columns
df_train.dtypes
df_train.describe()
df_train.isnull().sum() / df_train.shape[0]
# seaborn 활용한 결측치 시각화

sns.heatmap(df_train.isnull(), cbar=False)
# Missingno

msno.matrix(df_train)
msno.bar(df_train, log=True)
msno.heatmap(df_train)
msno.dendrogram(df_train)
df_test.isnull().sum() / df_test.shape[0]
f, ax = plt.subplots(1, 2, figsize=(18, 8))



df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], 

                                             autopct='%1.1f%%',ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')



plt.show()
# pclass 그룹 별 데이터 카운트

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins=True)
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().plot.bar()
f, ax = plt.subplots(1,2, figsize=(18,8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived by Sex')



sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Survived by Sex')

plt.show()
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
print('제일 나이 많은 탑승객 : {:.1f} Years'.format(df_train['Age'].max()))

print('제일 어린 탑승객 : {:.1f} Years'.format(df_train['Age'].min()))

print('탑승객 평균 나이 : {:.1f} Years'.format(df_train['Age'].mean()))
fig, ax = plt.subplots(1, 1, figsize=(9, 5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
plt.figure(figsize=(8,6))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')



plt.xlabel('Age')

plt.title('Age distribution within classes')

plt.legend(['1st class', '2nd class', '3rd class'])
cummulate_survival_ratio = []

for i in range(1, 80):

    cummulate_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))



plt.figure(figsize=(7, 7))

plt.plot(cummulate_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.ylabel('Survival rate')

plt.xlabel('Range of Age(0~x)')

plt.show()
df_train['Embarked'].unique()
f, ax = plt.subplots(1,1, figsize=(7,7))

df_train[['Embarked','Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
f, ax = plt.subplots(2,2, figsize=(20,15))

sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0,0].set_title('(1) No. of passengers boarded')

sns.countplot('Embarked', hue='Sex', data=df_train, ax=ax[0,1])

ax[0,1].set_title('(2) Male-Female split for embarked')

sns.countplot('Embarked', hue='Survived', data=df_train, ax=ax[1,0])

ax[1,0].set_title('(3) Embarked / Survived')

sns.countplot('Embarked', hue='Pclass', data=df_train, ax=ax[1,1])

ax[1,1].set_title('(4) Embarked / Pclass')

plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
print('Maximum size of family', df_train['FamilySize'].max())

print('Maximum size of family', df_train['FamilySize'].min())
f,ax=plt.subplots(1, 3, figsize=(40,10))

sns.countplot('FamilySize', data=df_train, ax=ax[0])

ax[0].set_title('(1) No. Of Passengers Boarded', y=1.02)

sns.countplot('FamilySize', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('(2) Survived countplot depending on FamilySize',  y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title('(3) Survived rate depending on FamilySize',  y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
fig, ax = plt.subplots(1,1, figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_test.loc[df_test.Fare.isnull(), 'Fare'] = df_test['Fare'].mean()



df_train['Fare'] = df_train['Fare'].map(lambda i : np.log(i) if i > 0 else 0)

df_test['Fare'] = df_train['Fare'].map(lambda i : np.log(i) if i > 0 else 0)
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness : {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train['Cabin'].isnull().sum() / df_train.shape[0]
df_train['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')

df_test['Initial'] = df_train.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='summer_r')
df_train['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)



df_test['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],

                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
df_train.groupby('Initial').mean()
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
msno.bar(df_train)
msno.bar(df_test)
# Fill Null in Embarked

df_train['Embarked'].fillna('S', inplace=True)

msno.bar(df_train)
def category_age(x) :

    if x< 10:

        return 0

    elif x < 20:

        return 1

    elif x < 30:

        return 2

    elif x < 40:

        return 3

    elif x < 50:

        return 4

    elif x < 60:

        return 5

    elif x < 70:

        return 6

    else:

        return 7

    

df_train['Age_cat'] = df_train['Age'].apply(category_age)

df_test['Age_cat'] = df_test['Age'].apply(category_age)
df_train.groupby('Age_cat')['PassengerId'].count()
# 데이터의 수치화



df_train['Initial'] = df_train['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})

df_test['Initial'] = df_test['Initial'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Other': 4})



df_train['Embarked'] = df_train['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

df_test['Embarked'] = df_test['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})



df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1})

df_test['Sex'] = df_test['Sex'].map({'female': 0, 'male': 1})
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat', 'Age']] 



colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0,

           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

# 데이터 전처리

# one-hot enconding

df_train = pd.get_dummies(df_train, columns=['Initial'], prefix='Initial')

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix='Initial')
df_train
df_train = pd.get_dummies(df_train, columns=['Embarked'], prefix='Embarked')

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix='Embarked')
df_train.head()
# 필요없는 피쳐들을 버린다.

df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
df_train.head()
df_train.dtypes
# 모델 개발 및 학습

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
# 학습 데이터와 target label 분리

X_train = df_train.drop('Survived', axis=1).values

target_label = df_train['Survived'].values

X_test = df_test.values
X_train.shape, X_test.shape, target_label.shape
# train_test split for making validation set

X_tr, X_vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.2, random_state=2020)
X_tr.shape, X_vld.shape, y_tr.shape, y_vld.shape
# 랜덤포레스트 모델 생성 및 학습

model = RandomForestClassifier()

model.fit(X_tr, y_tr)

prediction = model.predict(X_vld)
print('샘플 갯수 {}, 정확도 {:.2f}%'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
from pandas import Series



feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
# keras를 사용한 NN 모델

from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.optimizers import Adam, SGD
nn_model = Sequential()

nn_model.add(Dense(32,activation='relu',input_shape=(14,)))

nn_model.add(Dropout(0.2))

nn_model.add(Dense(64,activation='relu'))

nn_model.add(Dropout(0.2))

nn_model.add(Dense(32,activation='relu'))

nn_model.add(Dropout(0.2))

nn_model.add(Dense(1,activation='sigmoid'))



Loss = 'binary_crossentropy'

nn_model.compile(loss=Loss,optimizer=Adam(),metrics=['accuracy'])

nn_model.summary()
history = nn_model.fit(X_tr, y_tr, batch_size=64, epochs=1000,

                      validation_data=(X_vld, y_vld), verbose=1)
hists = [history]

hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)

hist_df.index = np.arange(1, len(hist_df)+1)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))

axs[0].plot(hist_df.val_accuracy, lw=5, label='Validation Accuracy')

axs[0].plot(hist_df.accuracy, lw=5, label='Training Accuracy')

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
# 모델 예측 및 평가



submission = pd.read_csv('../input/2019-1st-ml-month-with-kakr/sample_submission.csv')

submission.head()
prediction = model.predict(X_test)

submission['Survived'] = prediction
submission.to_csv('my_first_submission.csv', index=False)
submission = pd.read_csv('../input/2019-1st-ml-month-with-kakr/sample_submission.csv')

prediction = nn_model.predict(X_test)

prediction = prediction > 0.5

prediction = prediction.astype(np.int)

prediction = prediction.T[0]
submission['Survived'] = prediction

submission.to_csv('my_nn_submission.csv', index=False)
# 앙상블 모델 테스트해보기



from sklearn.linear_model import LogisticRegression

from subprocess import check_output



from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



from sklearn import svm, neighbors

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier





# model1 = svm.LinearSVC

model2 = neighbors.KNeighborsClassifier()

model3 = RandomForestClassifier()

model4 = LogisticRegression()

model5 = LinearDiscriminantAnalysis()

model6 = DecisionTreeClassifier()

model7 = MLPClassifier()

model8 = ExtraTreesClassifier()

model9 = AdaBoostClassifier()

model10 = GradientBoostingClassifier()

model11 = XGBClassifier(Eta=0.2)





clf = VotingClassifier(estimators=[

                        ('knn', model2),

                        ('rfor', model3),

                        ('lo-r', model4),

                        ('li-dr', model5),

                        ('dtc', model6),

                        ('mlpc', model7),

                        ('etc', model8),

                        ('abc', model9),

                        ('gbc', model10),

                        ('XGBC', model11)])



clf.fit(X_tr, y_tr)



confidence = clf.score(X_vld, y_vld)

print('accuracy : ', confidence)

predictions = clf.predict(X_test)
submission['Survived'] = predictions

submission.to_csv('my_ensemble_submission.csv', index=False)
# 배깅 모델 사용하기



from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier



model1 = DecisionTreeClassifier(random_state=1).fit(X_tr, y_tr)

ensemble_1 = BaggingClassifier(base_estimator=DecisionTreeClassifier(),).fit(X_tr, y_tr)



print('단일 모델 : ', model1.score(X_vld, y_vld))

print('ensemble_1 : ', ensemble_1.score(X_vld, y_vld))
model11.fit(X_tr, y_tr)



print('XGB : ', model11.score(X_vld, y_vld))