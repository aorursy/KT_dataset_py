import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt

%matplotlib inline

import warnings # just for ignoring annoying warnings
warnings.filterwarnings('ignore')
TARGET = 'Survived'
ID = 'PassengerId'
dataset = pd.read_csv('../input/train.csv')
dataset.head()
desc = dataset.describe()
desc
numerical_columns = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]
for col in numerical_columns:
    dataset[col].hist()
    plt.title(col + " distribution")
    plt.show();
    
    st = dataset[[col, TARGET]].groupby(col, as_index=False).mean()
    plt.bar(st[col], st[TARGET])
    plt.title(col + " survival count")
    plt.show();

del st
numerical_dataset = numerical_columns + [TARGET]
numerical_dataset = dataset[numerical_dataset]
numerical_dataset.corr()
numerical_dataset = numerical_dataset.drop('Survived', axis=1)
numerical_dataset.head()
categorical_variables = [
    'Sex',
    'Embarked'
]
categorical_dataset = dataset[categorical_variables + [TARGET]]
categorical_dataset.head()
for col in categorical_dataset.columns:
    print(categorical_dataset[col].value_counts())
    print("\n")
categorical_dataset = categorical_dataset.drop('Survived', axis=1)
num_feat_dataset = pd.DataFrame()
num_feat_dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
num_feat_dataset['FamilySize'].hist()
cat_feat_dataset = pd.DataFrame()
cat_feat_dataset[TARGET] = dataset[TARGET]
cat_feat_dataset['IsAlone'] = np.ones(dataset.shape[0])
cat_feat_dataset['IsAlone'].loc[num_feat_dataset['FamilySize'] > 1] = 0
cat_feat_dataset['AgeBin'] = pd.qcut(dataset['Age'].fillna(dataset['Age'].mean()).astype(int), 5)
cat_feat_dataset['FareBin'] = pd.qcut(dataset['Fare'].fillna(dataset['Fare'].mean()).astype(int), 4)
cat_feat_dataset['Title'] = dataset['Name'].str.extract(r',\s([A-Z].*)\.')
cat_feat_dataset['Title'].value_counts()
conditions = [
    cat_feat_dataset['Title'] == 'Mr',
    cat_feat_dataset['Title'] == 'Mrs',
    cat_feat_dataset['Title'] == 'Miss',
    cat_feat_dataset['Title'] == 'Master',
    cat_feat_dataset['Title'] == 'Dr',
    cat_feat_dataset['Title'] == 'Rev'
]

choices = [6, 5, 4, 3, 2, 1]

cat_feat_dataset['Title'] = np.select(conditions, choices, default=0)
st = cat_feat_dataset[['Title', TARGET]].groupby('Title', as_index=False).mean()
plt.bar(st['Title'], st[TARGET]);
dataset['Cabin'][dataset['Cabin'].isna()].shape
cat_feat_dataset['HasCabinInfo'] = dataset['Cabin'].isnull()
st = cat_feat_dataset[['HasCabinInfo', TARGET]].groupby('HasCabinInfo', as_index=False).mean()
plt.bar(st['HasCabinInfo'], st[TARGET]);
cat_feat_dataset['Deck'] = dataset['Cabin'].str.slice(0,1)
cat_feat_dataset['Deck'].value_counts()
st = cat_feat_dataset[['Deck', TARGET]].groupby('Deck', as_index=False).mean()
plt.bar(st['Deck'], st[TARGET]);
conditions = [
    (cat_feat_dataset['Deck'] == 'A') | (cat_feat_dataset['Deck'] == 'B') | (cat_feat_dataset['Deck'] == 'C'),
    (cat_feat_dataset['Deck'] == 'D') | (cat_feat_dataset['Deck'] == 'E'),
    (cat_feat_dataset['Deck'] == 'F') | (cat_feat_dataset['Deck'] == 'G') | (cat_feat_dataset['Deck'] == 'T')
]

choices = [1, 2, 3]

cat_feat_dataset['Deck'] = np.select(conditions, choices, default=0)
cat_feat_dataset['CabinPos'] = dataset["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
cat_feat_dataset['CabinPos'].hist()
cat_feat_dataset['CabinPos'] = pd.qcut(cat_feat_dataset['CabinPos'], 3)
cat_feat_dataset['CabinPos'].value_counts()
cat_feat_dataset[['CabinPos', TARGET]].groupby('CabinPos', as_index=False).mean()
cat_feat_dataset[['CabinPos', 'Deck', TARGET]].groupby(['CabinPos', 'Deck'], as_index=False).mean()
cat_feat_dataset = cat_feat_dataset.drop(TARGET, axis=1)
cat_feat_dataset['CabinPos'] = cat_feat_dataset['CabinPos'].astype(str)
cat_feat_dataset.head()
numerical_dataset = pd.concat([numerical_dataset, num_feat_dataset], axis=1)
categorical_dataset = pd.concat([categorical_dataset, cat_feat_dataset], axis=1)
for col in categorical_dataset.columns:
    print(col)
    print(categorical_dataset[categorical_dataset[col].isna()].index)
for col in numerical_dataset.columns:
    print(col)
    print(numerical_dataset[numerical_dataset[col].isna()].index)
dataset[dataset['Age'].isna()].head()
numerical_dataset['Age'] = numerical_dataset['Age'].fillna(dataset['Age'].mean())
rows_to_drop = [
    61, 829
]
categorical_dataset = categorical_dataset.drop(categorical_dataset.index[rows_to_drop])
numerical_dataset = numerical_dataset.drop(numerical_dataset.index[rows_to_drop])
numerical_scaler = StandardScaler()
numerical_scaler.fit(numerical_dataset)
numerical_dataset = numerical_scaler.transform(numerical_dataset)
numerical_dataset = pd.DataFrame(numerical_dataset)
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(categorical_dataset)
categorical_dataset = one_hot_encoder.transform(categorical_dataset)
categorical_dataset = pd.DataFrame(categorical_dataset.toarray())
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
X = pd.concat([numerical_dataset, categorical_dataset], axis=1)
y = dataset[TARGET]
y = y.drop(y.index[rows_to_drop])
X = X.fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier()
gbc = GradientBoostingClassifier()
svc = SVC(probability=True)
ext = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier()
X_train.columns = range(X_train.columns.shape[0])
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
scores = []
names = []

for model in models:
    model_name = model.__class__.__name__
    
    model.fit(X_train, y_train)
    acc = cross_val_score(model, X_train, y_train, scoring = "accuracy", cv = 10)
    scores.append(acc.mean())
    names.append(model_name)
results = pd.DataFrame({
    'Model': names,
    'Score': scores
})
results = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
results.head(len(models))
final_model = VotingClassifier(
    estimators=[(model.__class__.__name__, model) for model in models],
    voting='soft'
)
final_model.fit(X_train, y_train)
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, file=f)
test_dataset = pd.read_csv('../input/test.csv')
test_numset = test_dataset[numerical_columns]
test_numset['FamilySize'] = test_numset['SibSp'] + test_numset['Parch'] + 1
test_catset = test_dataset[categorical_variables]
test_catset['IsAlone'] = np.ones(test_dataset.shape[0])
test_catset['IsAlone'].loc[test_numset['FamilySize'] > 1] = 0

test_catset['AgeBin'] = pd.qcut(test_numset['Age'].fillna(test_numset['Age'].mean()).astype(int), 5)
test_catset['FareBin'] = pd.qcut(test_numset['Fare'].fillna(test_numset['Fare'].mean()).astype(int), 4)

test_catset['Title'] = test_dataset['Name'].str.extract(r',\s([A-Z].*)\.')
conditions = [
    test_catset['Title'] == 'Mr',
    test_catset['Title'] == 'Mrs',
    test_catset['Title'] == 'Miss',
    test_catset['Title'] == 'Master',
    test_catset['Title'] == 'Dr',
    test_catset['Title'] == 'Rev'
]

choices = [6, 5, 4, 3, 2, 1]

test_catset['Title'] = np.select(conditions, choices, default=0)

test_catset['HasCabinInfo'] = test_dataset['Cabin'].isnull()

test_catset['Deck'] = test_dataset['Cabin'].str.slice(0,1)

conditions = [
    (test_catset['Deck'] == 'A') | (test_catset['Deck'] == 'B') | (test_catset['Deck'] == 'C'),
    (test_catset['Deck'] == 'D') | (test_catset['Deck'] == 'E'),
    (test_catset['Deck'] == 'F') | (test_catset['Deck'] == 'G') | (test_catset['Deck'] == 'T')
]

choices = [1, 2, 3]

test_catset['Deck'] = np.select(conditions, choices, default=0)

test_catset['CabinPos'] = test_dataset["Cabin"].str.slice(1,5).str.extract("([0-9]+)", expand=False).astype("float")
test_catset['CabinPos'] = pd.qcut(test_catset['CabinPos'], 3)
test_catset['CabinPos'] = test_catset['CabinPos'].astype(str)
test_numset = numerical_scaler.transform(test_numset)
test_numset = pd.DataFrame(test_numset)
test_catset = one_hot_encoder.transform(test_catset)
test_catset = pd.DataFrame(test_catset.toarray())
X = pd.concat([test_numset, test_catset], axis=1)
X = X.fillna(0)
X.columns = range(X.columns.shape[0])
y = final_model.predict(X)
y.shape, test_dataset['PassengerId'].shape
submission = pd.DataFrame({
    'PassengerId': test_dataset['PassengerId'],
    'Survived': y
})
submission.head()
submission.to_csv('./submission.csv', index=False)
