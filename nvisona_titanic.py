import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
train_df = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train_df, test]
# Data structure



train_df.head()

train_df.info()   # number of non-null datatypes for column

train_df.describe()    # Numerical variables

train_df.describe(include=['O'])   # Non-numerical variables

print(pd.DataFrame({'Total': train_df.isnull().sum(), 'Percentage': train_df.isnull().sum()/len(train_df)}))



test.head()

test.info()   # number of non-null datatypes for column

test.describe()    # Numerical variables

test.describe(include=['O'])   # Non-numerical variables

print(pd.DataFrame({'Total': test.isnull().sum(), 'Percentage': test.isnull().sum()/len(test)}))



cols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']



for col in cols:

    print(train_df[[col, 'Survived']].groupby([col], as_index=False).mean().sort_values(by='Survived', ascending=False), '\n')
# Data visualization



p = sns.FacetGrid(train_df, col='Survived')

p.map(plt.hist, 'Age', bins=20)



p = sns.FacetGrid(train_df, col='Survived')

p.map(plt.hist, 'Fare', bins=20)



p = sns.FacetGrid(train_df, col='Survived', row='Pclass')

p.map(plt.hist, 'Age', bins=20)



p = sns.FacetGrid(train_df, row='Embarked', size= 2.2, aspect = 1.6)

p.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

p.add_legend()



p = sns.FacetGrid(train_df, row='Embarked', col='Sex')

p.map(sns.barplot, 'Sex', 'Fare', alpha=.5)

p.add_legend()



sns.heatmap(train.corr(), linewidth=2, annot=True, cmap="YlGnBu")

plt.yticks(rotation=0)

plt.xticks(rotation=45)

plt.show()
#                Features Engineering



# Extracting extra features



for ds in combine:

    ds['Title'] = ds['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)



for ds in combine:

    ds['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)

    ds['Title'].replace('Mlle', 'Miss', inplace=True)

    ds['Title'].replace('Ms', 'Miss', inplace=True)

    ds['Title'].replace('Mme', 'Mrs', inplace=True)



for ds in combine:

    ds['FamilySize'] = ds['SibSp'] + ds['Parch'] + 1



# sns.countplot(x='FamilySize', hue='Survived', data=train_df) # justifies switching to alone/not alone



for ds in combine:

    ds.loc[ds['FamilySize']==1 ,'IsAlone'] = 1

    ds.loc[ds['FamilySize']!=1 ,'IsAlone'] = 0

    ds['IsAlone'] = ds['IsAlone'].astype(int)

    ds.drop('FamilySize', axis=1, inplace=True)

# Fill missing values



print(pd.DataFrame({'Total': train_df.isnull().sum(), 'Percentage': train_df.isnull().sum()/len(train_df)}))

print(pd.DataFrame({'Total': test.isnull().sum(), 'Percentage': test.isnull().sum()/len(test)}))



print(train_df[train_df['Embarked'].isnull()])

sns.boxplot(x='Embarked', y='Fare', hue='Sex', data=train_df[(train_df['Sex']=='female') & (train_df['Pclass']==1)]) # 'C' and 'S' equally likely

train_df.loc[train_df['Embarked'].isnull(), 'Embarked'] = train_df['Embarked'].mode().values[0]



print(test[test['Fare'].isnull()])

test.loc[test['Fare'].isnull(), 'Fare'] = test[(test['Pclass']==3) & (test['Sex']=='male') & (test['Embarked']=='S')]['Fare'].mode().values[0]



train_df['AgeGroup'] = pd.qcut(train_df['Age'], 4)

train_df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup')



for ds in combine:

    ds.loc[ds['Age']<=20.125, 'AgeRange'] = 0

    ds.loc[(ds['Age']>20.125) & (ds['Age']<=28.0), 'AgeRange'] = 1

    ds.loc[(ds['Age']>28.0) & (ds['Age']<=38.0), 'AgeRange'] = 2

    ds.loc[ds['Age']>38.0, 'AgeRange'] = 3





train_df.drop('AgeGroup', axis=1, inplace=True)





train_df['FareGroup'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean().sort_values(by='FareGroup')



for ds in combine:

    ds.loc[ds['Fare']<=7.91, 'FareRange'] = 0

    ds.loc[(ds['Fare']>7.91) & (ds['Fare']<=14.454), 'FareRange'] = 1

    ds.loc[(ds['Fare']>14.454) & (ds['Fare']<=31.0), 'FareRange'] = 2

    ds.loc[ds['Fare']>31.0, 'FareRange'] = 3

    # ds['Age'] = ds['Age'].astype(int)



train_df.drop('FareGroup', axis=1, inplace=True)
### Fill missing Age values using KNN



age_df = train_df.drop(['PassengerId', 'Survived', 'Age', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

age_df = age_df.dropna()



le = LabelEncoder()

cols = ['Sex', 'Title', 'Embarked']

for col in cols:

    age_df[col] = le.fit_transform(age_df[col])



l = int(len(age_df)*0.7)

train = age_df.iloc[:l,:]

cv = age_df.iloc[l:,:]



Y_train = train['AgeRange']

X_train = train.drop('AgeRange', axis=1)

Y_cv = cv['AgeRange']

X_cv = cv.drop('AgeRange', axis=1)





# Find best n_neighbors



nn = []

train_sc = []

cv_sc = []



for i in range(2,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, Y_train)

    nn.append(i)

    train_sc.append(knn.score(X_train, Y_train))

    cv_sc.append(knn.score(X_cv,Y_cv))



plt.plot(nn, train_sc, label='train')

plt.hold('on')

plt.plot(nn, cv_sc, label='test')

plt.show()

plt.legend()



# n_neighbors doesn't affect much, so set = 3



age_train = train_df.drop(['PassengerId', 'Survived', 'Age', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

age_test = test.drop(['PassengerId', 'Age', 'Name', 'Ticket', 'Cabin', 'Fare'], axis=1)

age_train.name='age_train'

age_test.name='age_test'

age_combine = [age_train, age_test]





for ds in age_combine:

    trn = ds.dropna()

    tst = ds[ds['AgeRange'].isnull()]

    Y_train = trn['AgeRange']

    X_train = trn.drop('AgeRange', axis=1)

    # Y_test = tst['AgeRange']

    X_test = tst.drop('AgeRange', axis=1)



    clf = KNeighborsClassifier(n_neighbors=3)

    clf.fit(X_train, Y_train)

    if ds.name == 'age_train':

        train_df.loc[train_df['AgeRange'].isnull(), 'AgeRange'] = clf.predict(X_test)

    elif ds.name == 'age_test':

        test.loc[test['AgeRange'].isnull(), 'AgeRange'] = clf.predict(X_test)

    else:

        print('Error')
# Removing useless features



for ds in combine:

    ds.drop(['Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin'], axis=1, inplace=True)

    ds['AgeRange'] = ds['AgeRange'].astype(int)

    ds['FareRange'] = ds['FareRange'].astype(int)
#               Prediction models

# Data preparation



train_df_mix.drop('PassengerId', axis=1, inplace=True)

train_df_mix = shuffle(train_df)

passengerId = test['PassengerId']

test.drop('PassengerId', axis=1, inplace=True)



Y_train_df = train_df_mix['Survived']

X_train_df = train_df_mix.drop('Survived', axis=1)



l = int(len(train_df_mix)*0.7)

train = train_df_mix.iloc[:l,:]

cv = train_df_mix.iloc[l:,:]

trcv = [train, cv]



Y_train = train['Survived']

X_train = train.drop('Survived', axis=1)

Y_cv = cv['Survived']

X_cv = cv.drop('Survived', axis=1)

X_test = test



scaler = StandardScaler()

scaler.fit(X_train)

Xtrain = scaler.transform(X_train)

Xcv = scaler.transform(X_cv)

Xtest = scaler.transform(X_test)



models = []

train_scores = []

cv_scores = []
# Random Forest



n_est = []

train_sc = []

cv_sc = []



for i in range(1,20):

    forest = RandomForestClassifier(n_estimators=i)

    forest.fit(Xtrain, Y_train)

    n_est.append(i)

    train_sc.append(forest.score(Xtrain, Y_train))

    cv_sc.append(forest.score(Xcv,Y_cv))



plt.plot(n_est, train_sc, label='train')

plt.hold('on')

plt.plot(n_est, cv_sc, label='test')

plt.show()

plt.legend()



forest = RandomForestClassifier(n_estimators=10)

forest.fit(Xtrain, Y_train)

print(forest.score(Xtrain, Y_train))

print(forest.score(Xcv,Y_cv))



models.append('Random Forest')

train_scores.append()



# SVC



svc = SVC()

params = {'C':range(1, 2500, 500),

          'gamma': [.01, .001, .0001]}



clf = GridSearchCV(svc, param_grid=params)

clf.fit(Xtrain, Y_train)

print('SVC train score: ', clf.score(Xtrain, Y_train))

print('SVC CV score: ', clf.score(Xcv,Y_cv))



# KNN



nn = []

train_sc = []

cv_sc = []



for i in range(2,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(Xtrain, Y_train)

    nn.append(i)

    train_sc.append(knn.score(Xtrain, Y_train))

    cv_sc.append(knn.score(Xcv,Y_cv))



plt.plot(nn, train_sc, label='train')

plt.hold('on')

plt.plot(nn, cv_sc, label='test')

plt.show()

plt.legend()





knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(Xtrain, Y_train)

print('KNN train score: ', knn.score(Xtrain, Y_train))

print('KNN CV score: ', knn.score(Xcv,Y_cv))



# Neural Network



mpl = MLPClassifier(solver='lbfgs')

mpl.fit(Xtrain, Y_train)

print('NN train score: ', mpl.score(Xtrain, Y_train))

print('NN CV score: ', mpl.score(Xcv,Y_cv))



# with tuning



alpha = 10.**-np.arange(1,7)

hid = range(50,300,50)



res_train = pd.DataFrame(columns=alpha, index=hid)

res_cv = pd.DataFrame(columns=alpha, index=hid)



for a in alpha:

    for h in hid:

        mpl = MLPClassifier(solver='lbfgs', alpha=a, hidden_layer_sizes=h)

        mpl.fit(Xtrain, Y_train)

        res_train[a][h] = mpl.score(Xtrain, Y_train)

        res_cv[a][h] = mpl.score(Xcv,Y_cv)
### SUBMISSION WITH KNN



Y_train_df = train_df['Survived']

X_train_df = train_df.drop('Survived', axis=1)



scaler = StandardScaler()

Xtrain = scaler.fit_transform(X_train_df)

Xtest = scaler.fit_transform(X_test)



knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_df, Y_train_df)

print('KNN train score: ', knn.score(X_train_df, Y_train_df))





Y_pred = knn.predict(Xtest)



res = pd.DataFrame({'PassengerId': passengerId,

                    'Survived': knn.predict(Xtest)})


