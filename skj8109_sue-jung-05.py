import pandas as pd

import numpy as np



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.info()
test.info()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
f, ax = plt.subplots(1, 2, figsize=(18, 8))



train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')



sns.countplot('Sex',hue='Survived',data=train, ax=ax[1])

ax[1].set_title('Sex: Survived')



# train['Survived'][train_df['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1,0],shadow=True, startangle=90)

 

# # 여성 생존 확률

# train['Survived'][train_df['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1,1],shadow=True, startangle=90)

 

# ax[0].set_title('Survived (male)')

# ax[1].set_title('Survived (female)')

 

plt.show()







f, ax = plt.subplots(1, 2, figsize=(18, 8))



# train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0,0], shadow=True)

# ax[0,0].set_title('Pie plot - Survived')

# ax[0,0].set_ylabel('')



# sns.countplot('Sex',hue='Survived',data=train, ax=ax[0,1])

# ax[0,1].set_title('Sex: Survived')



train['Survived'][train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0],shadow=True, startangle=90)

 

# 여성 생존 확률

train['Survived'][train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1],shadow=True, startangle=90)

 

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')

 

plt.show()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
f, ax=  plt.subplots(1, 3, figsize=(18, 8))



train['Survived'][train['Pclass']==1].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[0],shadow=True, startangle=90)

train['Survived'][train['Pclass']==2].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[1],shadow=True, startangle=90)

train['Survived'][train['Pclass']==3].value_counts().plot.pie(explode=[0,0.1], autopct='%1.1f%%', ax=ax[2],shadow=True, startangle=90)

 

ax[0].set_title('Pclass 1')

ax[1].set_title('Pclass 2')

ax[2].set_title('Pclass 3')



plt.show()

 

# pd.crosstab([train_df['Sex'], train_df['Survived']], train_df['Pclass']

#             , margins=True).style.background_gradient(cmap='summer_r')
train_and_test = [train, test]
# train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.')

# pd.crosstab(train['Title'], train['Sex'])
# train['Title'] = train['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

# train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

# train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')

# train['Title'] = train['Title'].replace('Mme', 'Mrs')

# train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

train.head(5)

test['Name'].unique()
for dataset in train_and_test:

    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.')





train.head(5)
test['Title'].unique()
pd.crosstab(train['Title'], train['Sex'])
pd.crosstab(test['Title'], test['Sex'])
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don','Dona', 'Dr', 'Jonkheer',

                                               'Major', 'Rev'], 'Other')

    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')

    dataset['Title'] = dataset['Title'].replace(['Countess','Lady','Sir','Master'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')







#     Mr', 'Mrs', 'Miss', 'Master', 'Ms', 'Col', 'Rev', 'Dr', 'Dona'

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
for dataset in train_and_test:

    dataset['Title'] = dataset['Title'].astype(str)
test['Title'].unique()
for dataset in train_and_test:

    dataset['Sex'] = dataset['Sex'].astype(str)
for dataset in train_and_test:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].astype(str)
for dataset in train_and_test:

    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)

    dataset['Age'] = dataset['Age'].astype(int)

    train['AgeBand'] = pd.cut(train['Age'], 5)

print (train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()) # Survivied ratio about Age Band
for dataset in train_and_test:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    dataset['Age'] = dataset['Age'].map( { 0: 'Child',  1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'} ).astype(str)
train
# train.loc[(train['Age'].isnull())&(train['Title']=='Master'),'Age'] = 5

# train.loc[(train['Age'].isnull())&(train['Title']=='Miss'),'Age'] = 22

# train.loc[(train['Age'].isnull())&(train['Title']=='Mr'),'Age'] = 33

# train.loc[(train['Age'].isnull())&(train['Title']=='Mrs'),'Age'] = 36

# train.loc[(train['Age'].isnull())&(train['Title']=='Rare'),'Age'] = 45

# train.loc[(train['Age'].isnull())&(train['Title']=='Royal'),'Age'] = 43
# train_df['AgeGroup'] = 0

# train_df.loc[ train_df['Age'] <= 7, 'AgeGroup'] = 0

# train_df.loc[(train_df['Age'] > 7) & (train_df['Age'] <= 18), 'AgeGroup'] = 1

# train_df.loc[(train_df['Age'] > 18) & (train_df['Age'] <= 30), 'AgeGroup'] = 2

# train_df.loc[(train_df['Age'] > 30) & (train_df['Age'] <= 40), 'AgeGroup'] = 3

# train_df.loc[(train_df['Age'] > 40) & (train_df['Age'] <= 60), 'AgeGroup'] = 4

# train_df.loc[ train_df['Age'] > 60, 'AgeGroup'] = 5

# f,ax=plt.subplots(1,1,figsize=(10,10))

# sns.countplot('AgeGroup',hue='Survived',data=train_df, ax=ax)

# plt.show()
for dataset in train_and_test:

    dataset['Fare'] = dataset['Fare'].fillna(13.675) 
for dataset in train_and_test:

    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset in train_and_test:

    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]

    dataset['Family'] = dataset['Family'].astype(int)
train
train.info()

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from sklearn import metrics
train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin','AgeBand'], axis=1, inplace=True)

test.drop(['PassengerId', 'Name',  'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
train = pd.get_dummies(train)

test = pd.get_dummies(test)





# target_label = train['Survived'].values

# X_train = train.drop('Survived', axis=1)

# test_data = test.values



# X_tr, X_Vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)



# model = RandomForestClassifier()

# model.fit(X_tr, y_tr)

# prediction = model.predict(X_Vld)

# print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))


target_label = train['Survived'].values

X_train = train.drop('Survived', axis=1)

test_data = test.values



X_tr, X_Vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)



model =LogisticRegression()

model.fit(X_tr, y_tr)

prediction = model.predict(X_Vld)

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))


target_label = train['Survived'].values

X_train = train.drop('Survived', axis=1)

test_data = test.values



X_tr, X_Vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)



model = SVC()

model.fit(X_tr, y_tr)

prediction = model.predict(X_Vld)

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))


# target_label = train['Survived'].values

# X_train = train.drop('Survived', axis=1)

# test_data = test.values



# X_tr, X_Vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)



# model = KNeighborsClassifier()

# model.fit(X_tr, y_tr)

# prediction = model.predict(X_Vld)

# print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
# target_label = train['Survived'].values

# X_train = train.drop('Survived', axis=1)

# test_data = test.values



# X_tr, X_Vld, y_tr, y_vld = train_test_split(X_train, target_label, test_size=0.3, random_state=2018)



# model = GaussianNB()

# model.fit(X_tr, y_tr)

# prediction = model.predict(X_Vld)

# print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
test.info()
train.info()
submission = pd.read_csv('../input/sample_submission.csv')
prediction = model.predict(test_data)

submission['Survived'] = prediction
submission.to_csv('./submission.csv', index=False)