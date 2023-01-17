# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

print (df_train.info())
print (df_train.head())
df_train['sex_female'] = df_train['Sex'].apply(lambda x: 1 if x=='female' else 0)

df_train['age_snr'] = df_train['Age'].apply(lambda x: 1 if x >= 50 else 0)

df_train['age_mid'] = df_train['Age'].apply(lambda x: 1 if (x > 10 and x < 50) else 0)

df_train['age_jnr'] = df_train['Age'].apply(lambda x: 1 if x <= 10 else 0)

df_train['known_age'] = df_train['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)

df_train.loc[df_train['known_age'] == 0, 'age_mid'] = 1
print (df_train.head())
print (df_train.describe())
train = df_train.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)

fig = plt.subplots(figsize=(20,10))

sns.heatmap(train.astype(float).corr(), annot=True, cmap='plasma') # my daugther's favourite color
from sklearn.cross_validation import train_test_split

X = df_train.loc[:, ['PassengerId', 'Pclass', 'sex_female', 'age_jnr', 'known_age']]

y = df_train['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(

X, y, test_size = 0.3, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_valid_std = sc.transform(X_valid)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_valid_std)

print ((y_valid != y_pred).sum())

print (accuracy_score(y_valid, y_pred))
df_test = pd.read_csv('../input/test.csv')

print (df_test.info())

print (df_test.head())
df_test['sex_female'] = df_test['Sex'].apply(lambda x: 1 if x=='female' else 0)

df_test['age_snr'] = df_test['Age'].apply(lambda x: 1 if x >= 50 else 0)

df_test['age_mid'] = df_test['Age'].apply(lambda x: 1 if (x > 10 and x < 50) else 0)

df_test['age_jnr'] = df_test['Age'].apply(lambda x: 1 if x <= 10 else 0)

df_test['known_age'] = df_test['Age'].apply(lambda x: 0 if pd.isnull(x) else 1)

df_test.loc[df_test['known_age'] == 0, 'age_mid'] = 1

print (df_test.head())
test = df_test.loc[:, ['PassengerId', 'Pclass', 'sex_female', 'age_jnr', 'known_age']]

test_std = sc.transform(test)

yy_test = lr.predict(test_std)
df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                                                 'Survived': yy_test})

print (df_submission.head())

df_submission.to_csv('accountant_titanic_01.csv', index=False)
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB
classif_list = [SVC(kernel='linear', C=0.025), SVC(gamma=2, C=1), KNeighborsClassifier(10),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    AdaBoostClassifier(), GaussianNB()]
def classif_func(data_train, label_train, data_valid, label_valid, classif):

    classif.fit(data_train, label_train)

    y_pred = classif.predict(data_valid)

    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)
for classif in classif_list:

    print (classif, classif_func(X_train_std, y_train, X_valid_std, y_valid, classif))
def classif_Kne(data_train, label_train, data_valid, label_valid, k):

    classif = KNeighborsClassifier(k)

    classif.fit(data_train, label_train)

    y_pred = classif.predict(data_valid)

    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)
for i in range(1,30):

    print (i, classif_Kne(X_train_std, y_train, X_valid_std, y_valid, i))
def classif_SVC(data_train, label_train, data_valid, label_valid, k):

    classif = SVC(gamma=2, C=k)

    classif.fit(data_train, label_train)

    y_pred = classif.predict(data_valid)

    return ((label_valid != y_pred).sum()), accuracy_score(label_valid, y_pred)
for i in range(1,30):

    print (i, classif_SVC(X_train_std, y_train, X_valid_std, y_valid, i))
classif = KNeighborsClassifier(20)

classif.fit(X_train_std, y_train)

yy_test = classif.predict(test_std)

df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                                                 'Survived': yy_test})

print (df_submission.head())

df_submission.to_csv('accountant_titanic_02.csv', index=False)
classif = SVC(gamma=2, C=3)

classif.fit(X_train_std, y_train)

yy_test = classif.predict(test_std)

df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                                                 'Survived': yy_test})

print (df_submission.head())

df_submission.to_csv('accountant_titanic_03.csv', index=False)
classif = SVC(probability=True)

classif.fit(X_train_std, y_train)

yy_test = classif.predict(test_std)

df_submission = pd.DataFrame({'PassengerId': test['PassengerId'],

                                                 'Survived': yy_test})

print (df_submission.head())

df_submission.to_csv('accountant_titanic_04.csv', index=False)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_tr1 = df_train.copy()

df_te1 = df_test.copy()

dataset = [df_tr1, df_te1]
for data in dataset:

    data['Title'] = data['Name'].str.split(",", expand=True)[1].str.split(".", expand=True)[0]

    data['Title_count'] = data.groupby('Title')['Title'].transform('count')

    data['Title'].loc[data['Title_count'] <= 10] = 'Misc'

    data['Age'].fillna(data['Age'].median(), inplace=True)

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    data['IsAlone'] = 1

    data['IsAlone'].loc[data['FamilySize'] > 1] = 0

    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    data['Fare'].fillna(data['Fare'].median(), inplace=True)

    
for data in dataset:

    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)

    data['FareBin'] = pd.qcut(data['Fare'], 4)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for data in dataset:

    data['Sex_code'] = encoder.fit_transform(data['Sex'])

    data['Pclass_code'] = encoder.fit_transform(data['Pclass'])

    data['Title_code'] = encoder.fit_transform(data['Title'])

    data['Age_code'] = encoder.fit_transform(data['AgeBin'])

    data['Fare_code'] = encoder.fit_transform(data['FareBin'])

    data['Embarked_code'] = encoder.fit_transform(data['Embarked'])
for data in dataset:

    data.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 

               'Embarked', 'Title', 'Title_count', 'AgeBin', 'FareBin'], axis=1, inplace=True)
for data in dataset:

    print (data.info())
def model_classif(dataset, columnt_list, classif):

    X = dataset.loc[:, column_list]

    y = dataset['Survived']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)

    sc = StandardScaler()

    sc.fit(X_train)

    X_train_std = sc.transform(X_train)

    X_valid_std = sc.transform(X_valid)

    classif.fit(X_train_std, y_train)

    y_pred = classif.predict(X_valid_std)

    return (classif.__class__.__name__, (y_valid != y_pred).sum(), 

            accuracy_score(y_valid, y_pred), classif.get_params())
classif_list = [SVC(probability=True), 

                LogisticRegression(C=10.0, random_state=0), 

                KNeighborsClassifier(4), 

               GaussianNB()]

column_list = ['Pclass', 'FamilySize', 'Sex_code', 'Title_code',

                       'Age_code', 'Fare_code', 'Embarked_code']

for classif in classif_list:

    print (model_classif(df_tr1, column_list, classif))

    print ('----------------')
X = df_tr1.loc[:, column_list]

y = df_tr1['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_valid_std = sc.transform(X_valid)

X = df_te1.loc[:, column_list]

X_std = sc.transform(X)
classif = SVC(probability=True)

classif.fit(X_train_std, y_train)

y_pred = classif.predict(X_std)

y_hat = pd.DataFrame({'PassengerId': df_te1['PassengerId'], 'Survived': y_pred})

print (y_hat.info())

print (classif.__class__.__name__, classif.get_params())

y_hat.to_csv('accountant_titanic_05.csv', index=False)
classif = KNeighborsClassifier(4)

classif.fit(X_train_std, y_train)

y_pred = classif.predict(X_std)

yy_hat = pd.DataFrame({'PassengerId': df_te1['PassengerId'], 'Survived': y_pred})

print (yy_hat.info())

print (classif.__class__.__name__, classif.get_params())

yy_hat.to_csv('accountant_titanic_06.csv', index=False)