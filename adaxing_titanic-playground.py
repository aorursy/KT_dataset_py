# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
data.shape, test_data.shape
data.head()
# distribution of numerical features
data.describe()
# 25% travel with their siblies and/or spouses
# >75% do not travel with their parents and/or children
# <1% bought fare with 512
# distribution of categorical features
data.describe(include=['O'])
# about 23.6% duplicates in ticket, so it might be someone bought a group of tickets with same number
# about 77% missing in Cabin column, and about 28% duplicates, so it might be few people live in same cabin
# only 2 null values in Embarked column
data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived')
# higher class has better survival rate
data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived')
# female has better survival rate
g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# smallest, oldest ones have better survival rate
g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Pclass')
# class 3 has highest died rate, class 1 has better surivival rate
g = sns.FacetGrid(data, col='Survived', row='Pclass')
g.map(plt.hist, 'Age')
g.add_legend()
# at higher class 1, survival distribution is nearly normal
# at lower level, smaller age has better surivival rate relatively
g = sns.FacetGrid(data, row='Embarked', aspect=1.6)
g.map(sns.pointplot,'Pclass', 'Survived', 'Sex', palette='deep')
g.add_legend()
# only at port C, male has better surivival rate
# female in general has better surivival rate 
g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Fare')
g.add_legend()
# paid high fare has better surivival rate
g = sns.FacetGrid(data, col='Survived')
g.map(plt.hist, 'Embarked')
g.add_legend()
# port S has highest survival rate, C is next 
g = sns.FacetGrid(data, col='Embarked')
g.map(plt.hist, 'Fare')
# port S is mainly from lower fare, port C is from relatively high fare, Q is within lowest
g = sns.FacetGrid(data, col='Embarked', row = 'Survived')
g.map(sns.barplot, 'Sex', 'Fare', ci=None)
g.add_legend()
# helper function as evaluation metrics
def check_null(df, n=30):
    return df.isnull().sum().sort_values(ascending=False).head(n)

def score_model(model, x_train, y_train,  x_valid, y_valid, threshold=False):
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)
    mae = mean_absolute_error(y_valid, prediction)
    mse = mean_squared_error(y_valid, prediction)
    score = {}
    # if need to tune threshold, turn on flag
    # specific number need to be tuned
    if threshold:
        threshold = 0.6
        while threshold < .7:
            preds = [1 if prediction[i] >= threshold else 0 for i in range(len(prediction))]
            # accuracy = TP+TN/TP+FP+FN+TN
            # precision = TP/TP+FP
            # recall = TP/TP+FN
            num_tp = 0
            num_fp = 0
            num_fn = 0
            num_tn = 0
            if not isinstance(y_valid, list):
                y_valid = y_valid.to_list()
            for i in range(len(preds)):
                # positive
                if preds[i] == 1 & preds[i] == y_valid[i]:
                    num_tp += 1
                elif preds[i] == 1 & preds[i] != y_valid[i]:
                    num_fp += 1
                # negative
                elif preds[i] == 0 & preds[i] == y_valid[i]:
                    num_tn += 1
                elif preds[i] == 0 & preds[i] != y_valid[i]:
                    num_fn += 1
            accuracy = (num_tp+num_tn)/len(preds)
            precision = num_tp/(num_tp+num_fp)
            recall = num_tp/(num_tp+num_fn)
            score[threshold] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'mae': mae, 'mse': mse}
            threshold += 0.02
    else:
        score['mae'] = mae
        score['mse'] = mse
    return score
# loading
data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
# shuffle data
data = data.reindex(np.random.permutation(np.arange(len(data))))
valid_fraction = 0.2
valid_size = int(len(data) * valid_fraction)
train_data = data[:-valid_size]
valid_data = data[-valid_size:]
if train_data['Survived'].mean() != valid_data['Survived'].mean():
    print(f'self splitting does not have same proportion on label: {train_data["Survived"].mean(), valid_data["Survived"].mean()}')
    y = data.Survived
    x = data.loc[:, data.columns != 'Survived'].copy()
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state = 0)
    print(f'by using sklearn library, we got {y_train.mean(), y_valid.mean()}')
# preprocessing missing value
train_col_has_null = [col for col in x_train.columns if x_train[col].isnull().any()]
valid_col_has_null = [col for col in x_valid.columns if x_valid[col].isnull().any()]
# remove rows that age col contain null value
# not work !!! 
# x_train = x_train[x_train['Age'].notna()]
# x_valid = x_valid[x_valid['Age'].notna()]
# y_train = y_train[x_train.index[x_train['Age'].notna()]]
# y_valid = y_valid[x_valid.index[x_valid['Age'].notna()]]

# test_data = test_data[test_data['Age'].notna()]
# remove unnecessary cols
unnece_col = ['Name', 'Ticket', 'Cabin', 'Embarked']
x_train = x_train.drop(unnece_col, axis=1)
x_valid = x_valid.drop(unnece_col, axis=1)
test_data = test_data.drop(unnece_col, axis=1)
# imputate to remaining col has null
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='most_frequent')
imputed_x_train = pd.DataFrame(my_imputer.fit_transform(x_train))
imputed_x_valid = pd.DataFrame(my_imputer.transform(x_valid))
imputed_test_data = pd.DataFrame(my_imputer.transform(test_data))
imputed_x_train.columns = x_train.columns
for col in imputed_x_train.columns:
    imputed_x_train[col] = imputed_x_train[col].astype(x_train[col].dtypes.name)
imputed_x_valid.columns = x_valid.columns
for col in imputed_x_valid.columns:
    imputed_x_valid[col] = imputed_x_valid[col].astype(x_valid[col].dtypes.name)
imputed_test_data.columns = test_data.columns
for col in imputed_test_data.columns:
    imputed_test_data[col] = imputed_test_data[col].astype(imputed_test_data[col].dtypes.name)
# preprocessing categorical value - sex 
from sklearn.preprocessing import LabelEncoder
cat_col = imputed_x_train.select_dtypes('object').columns.tolist()
label_x_train = imputed_x_train.copy()
label_x_valid = imputed_x_valid.copy()
label_test_data = imputed_test_data.copy()
label_encoder = LabelEncoder()
for col in cat_col:
    label_x_train[col] = label_encoder.fit_transform(imputed_x_train[col])
    label_x_valid[col] = label_encoder.transform(imputed_x_valid[col])
    label_test_data[col] = label_encoder.transform(imputed_test_data[col])
model = RandomForestRegressor(n_estimators=127, max_depth=10, random_state=0)
score_model(model, label_x_train, y_train,  label_x_valid, y_valid)
# loading
data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
PassengerId = test_data.PassengerId

train, valid= train_test_split(data, random_state = 0)
combine = [train, valid, test_data]
# drop Ticket, Cabin col as num of duplicates in ticket and plenty missing value in Cabin also duplicate
for dataset in combine:
    print('Before drop ', dataset.shape)
    dataset.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
    print('After drop ',dataset.shape)
# Adding new features to data 
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
# sorting new feature Title
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Mlle', 'Mme'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Mrs')
    title_dic = dict(zip(dataset.groupby('Title').PassengerId.count().index.tolist(), dataset.groupby('Title').PassengerId.count().tolist()))
    for i, v in title_dic.items():
        if v<7:
            dataset['Title'] = dataset['Title'].replace(i, 'Rare')
# we can drop Name, passenerageId cols
for dataset in combine:
    print(pd.crosstab(dataset['Title'], dataset['Sex']))
    print('-----')
# check details of dataset 
for dataset in combine:
    print(dataset.shape)
    print(dataset.Age.describe())
    print('-------')
check_null(train), check_null(valid), check_null(test_data)
test_data[test_data.Fare.isnull()]
# only test dataset Fare column has null value with mean value
test_data.Fare.fillna(test_data.Fare.median(), inplace=True)
check_null(test_data)
# fill null value for embarked by mode
for dataset in combine:
    most_freq = dataset.Embarked.mode()[0]
    dataset['Embarked'] = dataset['Embarked'].fillna(most_freq)
# fill null value in age co by filling mean of same title columns
for dataset in combine:
    mean_based_title = dict(zip(dataset.groupby('Title').Age.mean().index.tolist(), dataset.groupby('Title').Age.mean().tolist()
))
    for k,v in dataset[dataset['Age'].isnull()].Title.items():
        dataset['Age'] = dataset['Age'].fillna(mean_based_title[dataset.loc[k].Title])
check_null(train, 5), check_null(valid, 5), check_null(test_data, 5)
# create age band to narrow down age distribution 
for dataset in combine:
    plt.hist(dataset.Age, width=2)
    plt.legend(['train_dataset', 'valid_dataset', 'test_dataset'])
print(f'In train dataset min age is {train.Age.min()} and max age is {train.Age.max()}.\nIn valid dataset min age is {valid.Age.min()} and max age is {valid.Age.max()}.\nIn test dataset min age is {test_data.Age.min()} and max age is {test_data.Age.max()}')
# it is from previous idea to create new feature AgeBand 
# train[['AgeBand', 'Survived']].groupby(['AgeBand']).mean().sort_values(by='Survived')
# valid[['AgeBand', 'Survived']].groupby(['AgeBand']).mean().sort_values(by='Survived')
# add new feature
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 
train[['FamilySize', 'Survived']].groupby(['FamilySize']).mean().sort_values(by='Survived')
valid[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived')
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='IsAlone')
for dataset in combine:
    plt.hist(dataset.Fare)
    plt.legend(['train_dataset', 'valid_dataset', 'test_dataset'])
# for dataset in combine:
#     dataset['FareBand'] = pd.cut(dataset.Fare, 4)
# train[['FareBand', 'Survived']].groupby('FareBand', as_index=False).mean().sort_values(by='FareBand')
train_y = train.Survived
valid_y = valid.Survived
train_x = train.loc[:, train.columns!='Survived'].copy()
valid_x = valid.loc[:, valid.columns!='Survived'].copy()
combine_1 = [train_x, valid_x, test_data]
for dataset in combine_1:
    print('Before drop: ', dataset.shape)
    dataset.drop(columns=['PassengerId', 'Name', 'SibSp', 'Parch'], inplace=True)
    print('After drop: ', dataset.shape)
# label_cat = ['FareBand']
# for feature in label_cat:
#     label_encoder = preprocessing.LabelEncoder()
#     train_x[feature] = label_encoder.fit_transform(train_x[feature])
#     valid_x[feature] = label_encoder.transform(valid_x[feature])
#     test_data[feature] = label_encoder.transform(test_data[feature])
train_x.groupby('Title').count()
for dataset in combine_1:
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    dataset['Title'] = dataset['Title'].map({'Master': 0, 'Miss': 1, 'Mr': 2, 'Mrs': 3, 'Rare': 4}).astype(int)
train_x
train_x.shape, valid_x.shape, test_data.shape
n_est = [i for i in range(50, 300, 25)]
scores = []
for i in n_est:
    model = RandomForestClassifier(n_estimators=i, random_state=0)
    model.fit(train_x, train_y)
    prediction = model.predict(valid_x)
    mae = mean_absolute_error(valid_y, prediction)
    scores.append(np.mean(mae))
plt.plot(n_est, scores)
# best score model n_est=130
model = RandomForestRegressor(n_estimators=100, random_state=0)
score_model(model, train_x, train_y, valid_x, valid_y).keys()
thresholds = []
scores = []
for k, v in score_model(model, train_x, train_y, valid_x, valid_y, True).items():
    thresholds.append(k)
    scores.append(np.mean(v['precision']))
plt.plot(thresholds, scores)
# best score model and threshold value=0.63 if using RandomForestRegressor
model = RandomForestRegressor(n_estimators=127, random_state=0)
score_model(model, train_x, train_y, valid_x, valid_y).keys()
thresholds = []
scores = []
for k, v in score_model(model, train_x, train_y, valid_x, valid_y, True).items():
    thresholds.append(k)
    scores.append(np.mean(v['precision']))
plt.plot(thresholds, scores)
# best score model and threshold value=0.62 and n_est=127 if using RandomForestRegressor
depth_ = [i for i in range(4, 12)]
scores = []
threshold = 0.62
for depth in depth_:
    model = RandomForestRegressor(n_estimators=127, max_depth=depth, random_state=0)
    model.fit(train_x, train_y)
    prediction = model.predict(valid_x)
    preds = [1 if prediction[i] >= threshold else 0 for i in range(len(prediction))]
    mae = mean_absolute_error(valid_y, preds)
    scores.append(np.mean(mae))
plt.plot(depth_, scores)
# best score model and threshold value=0.62 and n_est=127 and depth=10 if using RandomForestRegressor
# pick model 
model = RandomForestRegressor(n_estimators=127, max_depth=5, random_state=0)
model.fit(train_x, train_y)
prediction = model.predict(test_data)
threshold = 0.62
preds = [1 if prediction[i] >= threshold else 0 for i in range(len(prediction))]
pd.DataFrame({'PassengerId': PassengerId, 'Survived': preds}).to_csv('titanic_test7.csv', index=False)
# loading
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')
check_null(train_data, 5), check_null(test_data, 5)
train_data['Age'] = train_data['Age'].fillna(-0.5)
test_data['Age'] = test_data['Age'].fillna(-0.5)
bins = [-1, 0, 3, 10, 18, 25, 45, 60, np.inf]
labels = ['unknown', 'baby', 'child', 'teenager', 'young adult', 'adult', 'middle age', 'senior']
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins, labels=labels)
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins, labels=labels)
sns.barplot(x='AgeGroup', y='Survived', data=train_data)
plt.show()
# plp without cabin value survive rate is relatively low
train_data['CabinBool'] = train_data['Cabin'].notnull().astype('int')
test_data['CabinBool'] = test_data['Cabin'].notnull().astype('int')
sns.barplot(x='CabinBool', y='Survived', data=train_data)
plt.show()
y_train = train_data.pop('Survived')
# combin train dataset and test dataset
all_data = [train_data, test_data]
# title = pd.DataFrame()
title_dict = {
    'Capt': 'Officer',
    'Col': 'Officer',
    'Don': 'Rare',
    'Dona': 'Rare',
    'Dr': 'Officer',
    'Jonkheer': 'Rare',
    'Lady': 'Miss',
    'Major': 'Officer',
    'Master': 'Officer',
    'Miss': 'Miss',
    'Mlle': 'Miss',
    'Mme': 'Mrs',
    'Mr': 'Mr',
    'Mrs': 'Mrs',
    'Ms': 'Mrs',
    'Rev': 'Officer',
    'Sir': 'Mr',
    'the Countess': 'Rare'
}
for data in all_data:
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    data['Title'] = data['Title'].map(title_dict)

# title['Title'] = title.Title.map(title_dict)
# title = pd.get_dummies(title.Title)
for data in all_data:
#     data = pd.concat((data, title), axis=1)
    data.pop('Name')
# Cabin col become N, C, S, 
for data in all_data:
    data['Cabin'] = data['Cabin'].fillna('NA')
    data['Cabin'] = data['Cabin'].map(lambda s:s[0])
    data.pop('Ticket')
# PCLASS change datatype to string as categorical data, later transfer to one-hot
for data in all_data:
    data['Pclass'] = data['Pclass'].astype(str)
    data.isnull().sum().sort_values(ascending=False)
# fill null value 
for data in all_data:
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data.isnull().sum().sort_values(ascending=False)
for data in all_data:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
train_data = pd.concat([train_data, pd.get_dummies(train_data[['Pclass', 'Sex', 'Embarked', 'Cabin', 'AgeGroup', 'Title']])], axis=1)
test_data = pd.concat([test_data, pd.get_dummies(test_data[['Pclass', 'Sex', 'Embarked', 'Cabin', 'AgeGroup', 'Title']])], axis=1)
train_data.shape
# encode cat features
for data in all_data:
#     feature_dummies = pd.get_dummies(data[['Pclass', 'Sex', 'Embarked', 'Cabin', 'AgeGroup', 'Title']])
#     data = pd.concat([data, pd.get_dummies(data[['Pclass', 'Sex', 'Embarked', 'Cabin', 'AgeGroup', 'Title']])], axis=1)
    # drop cols
    print('Before: ', data.shape)
    data.drop(['Pclass', 'Sex', 'Embarked', 'Cabin','AgeGroup', 'Age', 'SibSp', 'Parch'], inplace=True, axis=1)
    print('After: ', data.shape)
#     data = pd.concat((data, feature_dummies), axis=1)
train_data.drop(['Pclass', 'Sex', 'Embarked', 'Cabin','AgeGroup', 'Age', 'SibSp', 'Parch'], inplace=True, axis=1)
test_data.drop(['Pclass', 'Sex', 'Embarked', 'Cabin','AgeGroup', 'Age', 'SibSp', 'Parch'], inplace=True, axis=1)
# split dataset 
# train_df = all_data.iloc[train_data.index]
# test_df = all_data.iloc[test_data.index]
# train_df.shape, test_df.shape
train_df=train_data
test_df=test_data
train_df.drop('Title', axis=1, inplace=True)
test_df.drop('Title', axis=1, inplace=True)
# model train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
depth_ = [i for i in range(1, 9)]
scores = []
for depth in depth_:
    clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=0)
    test_score = cross_val_score(clf, train_df, y_train, cv=5, scoring='precision')
    scores.append(np.mean(test_score))
plt.plot(depth_, scores)
# depth at 7 reaches max arount 0.82 with cv=5
# depth at 2 reaches max arount 0.75 with cv=5
n_est = [i for i in range(25,200,20)]
scores = []
for n in n_est:
    clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
    test_score = cross_val_score(clf, train_df, y_train, cv=5, scoring='precision')
    scores.append(np.mean(test_score))
plt.plot(n_est, scores)
# n_estimators at 130 reach 0.826 with cv=5
# n_estimators at 60 reach 0.74 with cv=5
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(train_df, y_train,random_state=0)
train_x.shape, valid_x.shape, test_data.shape
train_x.drop(['Cabin_T'], axis=1, inplace=True)
valid_x.drop(['Cabin_T'], axis=1, inplace=True)
# as train dataset in cabin col has start with T with only one row, however test dataset doesn't have just delete from trainset
from sklearn.metrics import mean_absolute_error
n_est = [i for i in range(25,150,10)]
scores = []
for n in n_est:
    clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
    clf.fit(train_x, train_y)
    predition = clf.predict(valid_x)
    scores.append(np.mean(mean_absolute_error(valid_y, predition)))
plt.plot(n_est, scores)
# n_est=75 reach lowest 
# n_est=45 reach lowest 
depth_ = [i for i in range(5, 20)]
scores = []
for depth in depth_:
    clf = RandomForestClassifier(n_estimators=45, max_depth=depth, random_state=0)
    clf.fit(train_x, train_y)
    predition = clf.predict(valid_x)
    scores.append(np.mean(mean_absolute_error(valid_y, predition)))
plt.plot(depth_, scores)
# depth=6 reach lowest
# depth=11 reach lowest 0.165
clf = RandomForestClassifier(n_estimators=45, max_depth=11, random_state=0)
clf.fit(train_x, train_y)
predition = clf.predict(valid_x)
mean_absolute_error(valid_y, predition)
final_clf = RandomForestClassifier(n_estimators=45, max_depth=11, random_state=0)
final_clf.fit(train_x, train_y)
result = final_clf.predict(test_df)
pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': result}).to_csv('titanic_test6.csv', index=False)
# get whole dataset
data = pd.read_csv('../input/titanic/train.csv')
y = data.Survived
test = pd.read_csv('../input/titanic/test.csv')
Id = test.PassengerId
model_on_full_data = RandomForestClassifier(n_estimators=75, max_depth=6, random_state=0)
model_on_full_data.fit(train_data, y)
preds = model_on_full_data.predict(test_data)
pd.DataFrame({'PassengerId': Id, 'Survived': preds}).to_csv('titanic_test8.csv', index=False)
