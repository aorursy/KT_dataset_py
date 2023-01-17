import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import accuracy_score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from tpot import TPOTClassifier
train_df = pd.read_csv('../input/titanic/train.csv') 

test_df = pd.read_csv('../input/titanic/test.csv')



print('train_df.shape:', train_df.shape)

print('test_df.shape:', test_df.shape)
# combine train, test data

train_test_data = [train_df, test_df]
print(train_df.columns.to_list())

print(test_df.columns.to_list())
train_df.info()
train_df.describe(include='object')
test_df.info()
test_df.describe(include='object')
# null field

train_null_s = train_df.isnull().sum()

print(train_null_s[train_null_s != 0])

print('-'*80)

test_null_s = test_df.isnull().sum()

print(test_null_s[test_null_s != 0])
# detect target outlier index

outlier_detection_field = ['Age', 'Fare']

weight = 2



outlier_indices = []



for col in outlier_detection_field:

    q1 = np.nanpercentile(train_df[col], 25)

    q3 = np.nanpercentile(train_df[col], 75)

    iqr = q3-q1

    iqr_weight = iqr * weight



    lowest_val = q1 - iqr_weight

    highest_val = q3 + iqr_weight



    outlier_index = train_df[(train_df[col]<lowest_val) | (highest_val<train_df[col])].index

    outlier_indices.extend(outlier_index)

    

    print('{}: {} / {} (record size:{})'.format(col, lowest_val, highest_val, outlier_index.shape[0]))
# drop outlier index

train_df.drop(outlier_indices, axis=0, inplace=True)
train_df['PassengerId']
test_df['PassengerId']
# drop 'passengerId' field

# for test data, save the 'PassengerId' field for submission



train_df.drop('PassengerId', axis=1, inplace=True)



test_df_PId = test_df['PassengerId']

test_df.drop('PassengerId', axis=1, inplace=True)
print(train_df.columns.to_list())

print(test_df.columns.to_list())
print(train_df['Pclass'].value_counts())



sns.barplot(data=train_df, x='Pclass', y='Survived')
# scale

minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Pclass'] = minMaxScaler.fit_transform(data[['Pclass']])
train_df['Pclass'].value_counts()
train_df['Name'].head(10)
# get 'title' field from 'name' field



# train_df['Name'].str.extract(' ([a-zA-Z]+)\. ', expand=False).value_counts()

for data in train_test_data:

    data['Title'] = data['Name'].str.extract(' ([a-zA-Z]+)\. ', expand=False)
# drop 'name' field



for data in train_test_data:

    data.drop('Name', axis=1, inplace=True)
print(train_df['Title'].value_counts())

print('-'*50)

print(test_df['Title'].value_counts())
# encode



title_mapping = {

    'Mr':0,

    'Miss':1,

    'Mrs':2,

    'Master':3,

    'Dr':4, 'Rev':4, 'Major':4, 'Mlle':4, 'Col':4, 'Ms':4, 'Countess':4, 'Mme':4, 'Lady':4, 'Sir':4, 'Don':4, 'Jonkheer':4, 'Capt':4, 'Dona':4

}



for data in train_test_data:

    data['Title'] = data['Title'].map(title_mapping)
train_df['Title'].value_counts()
# scale



minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Title'] = minMaxScaler.fit_transform(data[['Title']])
train_df['Title'].value_counts()
sns.barplot(data=train_df, x='Title', y='Survived')
print(train_df['Sex'].value_counts())



sns.barplot(data=train_df, x='Sex', y='Survived')
# encode



for data in train_test_data:

    data['Sex'] = data['Sex'].astype('category').cat.codes
train_df['Sex'].value_counts()
train_df['Age'].isnull().sum()
# fill null with the middle value of the title

# train_df.groupby('Title')['Age'].transform('median')



for data in train_test_data:

    data['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'), inplace=True)
train_df['Age'].isnull().sum()
sns.distplot(train_df['Age'])
# binning

# pd.qcut(train_df['Age'], 5).cat.codes



for data in train_test_data:

    data['Age'] = pd.qcut(data['Age'], 9).cat.codes
# scale



minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Age'] = minMaxScaler.fit_transform(data[['Age']])
train_df['Age'].describe()
print(train_df['SibSp'].value_counts())

sns.barplot(data=train_df, x='SibSp', y='Survived')
print(train_df['Parch'].value_counts())

sns.barplot(data=train_df, x='Parch', y='Survived')
for data in train_test_data:

    data['FamilySize'] = data['Parch'] + data['SibSp']
sns.barplot(data=train_df, x='FamilySize', y='Survived')
# drop

for data in train_test_data:

    data.drop(['SibSp', 'Parch'], axis=1, inplace=True)
# binning

# train_df.loc[(1<=train_df['FamilySize']) & (train_df['FamilySize']<4), 'FamilySize'].value_counts()



# for data in train_test_data:

#     data.loc[data['FamilySize']==0, 'FamilySize'] = 0

#     data.loc[(1<=data['FamilySize']) & (data['FamilySize']<4), 'FamilySize'] = 1

#     data.loc[(4<=data['FamilySize']) & (data['FamilySize']<7), 'FamilySize'] = 2

#     data.loc[(7<=data['FamilySize']), 'FamilySize'] = 3
# scale

minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['FamilySize'] = minMaxScaler.fit_transform(data[['FamilySize']])
sns.barplot(data=train_df, x='FamilySize', y='Survived')
print(train_df['Ticket'].value_counts())

print('-'*80)

print(train_df['Ticket'].unique().shape)
# drop 'Ticket' field

for data in train_test_data:

    data.drop('Ticket', axis=1, inplace=True)
print(train_df['Fare'].isnull().sum())

print(test_df['Fare'].isnull().sum())
train_df.groupby(['Embarked', 'Pclass'])['Fare'].median()
# fill null with the middle value of the 'Embarked', 'Pclass'



for data in train_test_data:

    data['Fare'].fillna(train_df.groupby(['Embarked', 'Pclass'])['Fare'].transform('median'), inplace=True)
test_df['Fare'].isnull().sum()
sns.distplot(train_df['Fare'])
# log transformation to import skewed data

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(2, 1, 1)

ax2 = fig.add_subplot(2, 1, 2)



sns.distplot(train_df['Fare'], ax=ax1)

sns.distplot(np.log1p(train_df['Fare']), ax=ax2)



for data in train_test_data:

    data['Fare'] = np.log1p(data[['Fare']])
# binning

# pd.qcut(train_df['Fare'], 5).astype('category').cat.codes.value_counts()



for data in train_test_data:

    data['Fare'] = pd.qcut(data['Fare'], 10).astype('category').cat.codes
train_df['Fare'].value_counts()
# scale

minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Fare'] = minMaxScaler.fit_transform(data[['Fare']])
train_df['Fare'].value_counts()
print(train_df['Cabin'].value_counts())

print('-'*80)

print(train_df['Cabin'].unique().shape)

print('-'*80)

print(train_df['Cabin'].str[:1].value_counts())
print(test_df['Cabin'].str[:1].value_counts())
# replace 'Cabin' field to the first character of the field

for data in train_test_data:

    data['Cabin'] = data['Cabin'].str[:1]
sns.barplot(data=train_df, x='Cabin', y='Survived')
# encode

cabin_mapping={"A":1, "B":2, "C":3, "D":4, "E":5, "F":6, "G":7, "T":8}



for data in train_test_data:

    data['Cabin'] = data['Cabin'].map(cabin_mapping)
print(train_df['Cabin'].value_counts())

print('-'*80)

# print(train_df.groupby(['Pclass', 'Embarked'])['Cabin'].median())

print(train_df.groupby(['Pclass'])['Cabin'].median())
# fill null with the middle value of the 'Pclass'

for data in train_test_data:

    data['Cabin'].fillna(data.groupby(['Pclass'])['Cabin'].transform('median'), inplace=True)
print(train_df['Cabin'].isnull().sum())

print('-'*80)

print(train_df['Cabin'].value_counts())
# scale

minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Cabin'] = minMaxScaler.fit_transform(data[['Cabin']])
print(train_df['Embarked'].isnull().sum())

print(test_df['Embarked'].isnull().sum())
print(train_df['Embarked'].value_counts())

sns.barplot(data=train_df, x='Embarked', y='Survived')
# there aren't many missing values(just 2 records in train data), so fill null to most value

for data in train_test_data:

    data['Embarked'] = data['Embarked'].fillna('S')
# encode



for data in train_test_data:

    data['Embarked'] = data['Embarked'].astype('category').cat.codes
# scale

minMaxScaler = MinMaxScaler()



for data in train_test_data:

    data['Embarked'] = minMaxScaler.fit_transform(data[['Embarked']])
train_df.head()
y_train_s = train_df['Survived']

x_train_df = train_df.drop('Survived', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_s, test_size=0.2, random_state=10)
def cross_val_score_result(estimator, x, y, scoring, cv):

    clf_scores = cross_val_score(estimator, x, y, scoring=scoring, cv=cv)

    clf_scores_mean = np.round(np.mean(clf_scores), 4)

    

    return clf_scores_mean
classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    KNeighborsClassifier(),

    SVC(),

    LGBMClassifier(),

    XGBClassifier(), 

    AdaBoostClassifier()

#     TPOTClassifier()

]
best_clf_score = 0

best_clf = None



clf_name = []

clf_mean_score = []



for clf in classifiers:

    current_clf_score = cross_val_score_result(clf, x_train, y_train, 'accuracy', 10)

    clf_name.append(clf.__class__.__name__)

    clf_mean_score.append(current_clf_score)

    

    if current_clf_score > best_clf_score:

        best_clf_score = current_clf_score

        best_clf = clf
clf_df = pd.DataFrame({"clf_name":clf_name, "clf_mean_score":clf_mean_score})

plt.figure(figsize=(8, 6))

sns.barplot(data=clf_df, x="clf_mean_score", y="clf_name")



print('best classifier: {}({})'.format(best_clf.__class__.__name__, best_clf_score))
# train the classifier get the highest score

lgbm_clf = LGBMClassifier()



grid_param = {

    'learning_rate':[0.005, 0.01, 0.015, 0.02],

    'n_estimators':[100, 150, 200],

    'bossting_type':['rf', 'gbdt', 'dart', 'goss'],

    'max_depth':[10, 15, 20]

}



lgbm_grid = GridSearchCV(lgbm_clf, grid_param, cv=10)

lgbm_grid.fit(x_train, y_train)
print('best_param:', lgbm_grid.best_params_)

print('best_score:{:.4f}'.format(lgbm_grid.best_score_))
test_df.head()
test_pred = lgbm_grid.best_estimator_.predict(test_df)



submission = pd.DataFrame({

    'PassengerId': test_df_PId,

    'Survived': test_pred

})
submission.to_csv('submission_test.csv', index=False)
check_submission = pd.read_csv('submission_test.csv')

check_submission