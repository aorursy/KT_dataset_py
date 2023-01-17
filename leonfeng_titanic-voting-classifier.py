import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('bmh')

%matplotlib inline

from sklearn import preprocessing



from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier



from sklearn.model_selection import cross_val_score, GridSearchCV
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



combine = [train, test]

for df in combine:

    print (df.info())

    print ('-'*50)
train.head(3)
train.describe()
train.describe(include=['O'])
train.pivot_table(values='Survived', columns='Sex', index=['Pclass'], aggfunc='mean').plot(kind='bar')

plt.xticks(rotation='0')
train.pivot_table(values='Survived', columns='Sex', index=['Embarked'], aggfunc='mean').plot(kind='bar')

plt.xticks(rotation='0')
plt.subplot2grid((1, 2), (0, 0))

train['SibSp'].value_counts().plot(kind='bar').legend()

plt.xticks(rotation='0')



plt.subplot2grid((1, 2), (0, 1))

train['Parch'].value_counts().plot(kind='bar').legend()

plt.xticks(rotation='0')
for df in combine:

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    

train['FamilySize'].value_counts()
for df in combine:

    df['Alone'] = 0

    df.loc[df['FamilySize'] == 1, 'Alone'] = 1

    

train['Alone'].value_counts()
for df in combine:

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    

train['Embarked'].value_counts()
for df in combine:

    df['Fare'].fillna(df['Fare'].median(), inplace=True)

    #df['Fare'] = scale(df['Fare'])

    df.loc[df['Fare'] <= 10.5, 'Fare'] = 0

    df.loc[(df['Fare'] > 10.5) & (df['Fare'] <= 21.679), 'Fare'] = 1

    df.loc[(df['Fare'] > 21.679) & (df['Fare'] <= 39.688), 'Fare'] = 2

    df.loc[(df['Fare'] > 39.688) & (df['Fare'] <= 512.329), 'Fare'] = 3

    df.loc[df['Fare'] > 512.329, 'Fare'] = 4 

    

train[['Fare', 'Survived']].groupby('Fare', as_index=False).mean()
for df in combine:

    avg = df['Age'].mean()

    std = df['Age'].std()

    NaN_count = df['Age'].isnull().sum()

    

    age_fill = np.random.randint(avg-std, avg+std, NaN_count)

    df.loc[df['Age'].isnull(), 'Age'] = age_fill

    df['Age'] = df['Age'].astype(int)

    

    df.loc[df['Age'] <= 16, 'Age'] = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[df['Age'] > 64, 'Age'] = 4

    

train[['Age', 'Survived']].groupby('Age').mean()
import re



def only_title(name):

    title = re.findall(' ([A-Za-z]+)\.', name)

    if title:

        return title[0]



for df in combine:

    df['Title'] = df['Name'].apply(only_title)

    

train['Title'].value_counts()
for df in combine:

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 

                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby('Title', as_index=False).mean()
train.head(2)
feature_drop = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']



for df in combine:

    df.drop(feature_drop, axis=1, inplace=True)



train.head(2)
def encode_features(train, test):

    features = ['Sex', 'Embarked', 'Age', 'Title']

    df_combined = pd.concat([train[features], test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        train[feature] = le.transform(train[feature])

        test[feature] = le.transform(test[feature])

    return train, test

    

train, test = encode_features(train, test)

train.head()
colormap = plt.cm.bone_r

plt.figure(figsize=(10,10))

plt.title('Correlations of Features', y=1.04, size=20)

sns.heatmap(train.astype(float).corr(), square=True, cmap=colormap, annot=True, linewidth=0.2)
X = train.drop(['Survived'], axis=1)

y = train['Survived']



print('X', X.shape)

print('y', y.shape)

print('Null Accuracy for y_test dataset:', y.value_counts()[1]/len(y))
def model_search(model, param_grid):

    grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')

    grid_fit = grid.fit(X, y)

    best_model = grid_fit.best_estimator_

    test_score = grid_fit.best_score_

    return best_model, test_score
svc = SVC(gamma='auto', probability=True)

svc_params = {'C': np.logspace(-2, 3, 6)}



logreg = LogisticRegression()

logreg_params = {'C': np.logspace(-2, 3, 6)}



rf = RandomForestClassifier(max_features='auto')

rf_params = {'n_estimators': list(range(10, 110, 10)), 'criterion':['gini', 'entropy']}



knn = KNeighborsClassifier()

knn_params = {'n_neighbors':list(range(10, 110, 10)), 'weights':['distance', 'uniform']}



et = ExtraTreesClassifier(max_features='auto')

et_params = {'n_estimators': list(range(10, 110, 10)), 'criterion':['gini', 'entropy']}



gb = GradientBoostingClassifier(max_features='auto')

gb_params = {'n_estimators': list(range(10, 110, 10))}
print(model_search(svc, svc_params), '\n')

print(model_search(logreg, logreg_params), '\n')

print(model_search(rf, rf_params), '\n')

print(model_search(knn, knn_params), '\n')

print(model_search(et, et_params), '\n')

print(model_search(gb, gb_params))
vclf = VotingClassifier(estimators=[('svc', svc), ('rf', rf), ('gb', gb)])

param_grid = {'voting':['hard', 'soft']}

grid = GridSearchCV(vclf, param_grid, cv=10, n_jobs=-1)

grid_fit = grid.fit(X, y)

grid_fit.best_score_
vclf = grid_fit.best_estimator_

vclf.fit(X, y)

pred = vclf.predict(test)



test_id = pd.read_csv('../input/test.csv')['PassengerId']

output = pd.DataFrame({'PassengerId' : test_id, 'Survived': pred})



output.to_csv('Predictions.csv', index = False)

output.head()