import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
print(os.listdir("../input"))
pd.set_option('display.expand_frame_repr', False)
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('Train shape:', train.shape)
print('Test shape:', test.shape)
print(train.columns.values)
train.describe(include = 'all')
print(train.dtypes)
print()
#Explore Nan values in each column
print(train.isna().sum())
print(test.dtypes)
print()
print(test.isna().sum())
# Check the correlation for the current numeric feature set.
print(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())
sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(),\
            annot = True, fmt = ".2f", cmap = "coolwarm")
mask = np.zeros_like(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), \
            annot = True, fmt = ".2f", cmap = "coolwarm", mask = mask)
print(train.columns.values)
# the relation between Pclass and Survived and mouthache boxes
print(train[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x = 'Pclass', y = 'Survived',  kind = 'bar', data = train)
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x = 'Sex', y = 'Survived',  kind = 'bar', data = train)
palette = 'viridis'
for sex in ('male', 'female'):
    tr = train[train['Sex'] == sex]
    age_bins = pd.qcut(tr['Age'], 20)
    df = tr.groupby(age_bins)['Survived'].mean()
    
    plt.figure(figsize=(12,4)).suptitle(f'Survived wrt. Age ({sex})', fontsize=15);
    sns.barplot(df.index, df.values, palette = palette).set_xticklabels(df.index, rotation=90);
sns.catplot(x = 'Sex', y = 'Survived',  kind = 'bar', data = train, hue = 'Pclass')
f = sns.FacetGrid(train, col = 'Survived')
f = f.map(sns.distplot, "Fare")
group = pd.cut(train.Fare, [0, 50, 100, 150, 200, 550])
piv_fare = train.pivot_table(index = group, columns = 'Survived', values = 'Fare', aggfunc = 'count')
piv_fare.plot(kind = 'bar')
ag = sns.FacetGrid(train, col = 'Survived')
ag = ag.map(sns.distplot, "Age")
group = pd.cut(train.Age, [0, 14, 30, 60, 100])
piv_fare = train.pivot_table(index = group, columns = 'Survived', values = 'Age', aggfunc = 'count')
piv_fare.plot(kind = 'bar')
print(train[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x = 'Embarked', y = 'Survived',  kind = 'bar', data = train)
sns.catplot('Pclass', kind = 'count', col = 'Embarked', data = train)
print(train[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x = 'SibSp', y = 'Survived', data = train, kind = 'bar')
print(train[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x = 'Parch', y = 'Survived', data = train, kind = 'bar')
print(train.Name.head(1))
print(train.Name.head(1).str.split(','))
for dataset in [train, test]:
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    print(dataset['Title'].value_counts())
    print()
sns.catplot(x = 'Survived', y = 'Title', data = train, kind = 'bar')
print(train[['Ticket', 'Survived']].groupby(['Ticket']).mean())
sns.catplot(x = 'Ticket', y = 'Survived',  kind = 'bar', data = train)
for df in [train, test]:
    print(df.shape)
    print()
    print(df.isna().sum())
# Drop rows with nulls for Embarked
for df in [train, test]:
    df.dropna(subset = ['Embarked'], inplace = True)
print(train[train['Fare'].isnull()])

print(test[test['Fare'].isnull()])
# We can deduce that Pclass should be related to Fares.
sns.catplot(x = 'Pclass', y = 'Fare', data = test, kind = 'point')
# There is a clear relation between Pclass and Fare. We can use this information to impute the missing fare value.
# We see that the passenger is from Pclass 3. So we take a median value for all the Pclass 3 fares.
test['Fare'].fillna(test[test['Pclass'] == 3].Fare.median(), inplace = True)
print(train[['Age','Title']].groupby('Title').mean())
sns.catplot(x = 'Age', y = 'Title', data = train, kind = 'bar')
def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
print(getTitle(train[train.Age.isnull()].Name).value_counts())

mr_mask = train['Title'] == 'Mr'
miss_mask = train['Title'] == 'Miss'
mrs_mask = train['Title'] == 'Mrs'
master_mask = train['Title'] == 'Master'
dr_mask = train['Title'] == 'Dr'
train.loc[mr_mask, 'Age'] = train.loc[mr_mask, 'Age'].fillna(train[train.Title == 'Mr'].Age.mean())
train.loc[miss_mask, 'Age'] = train.loc[miss_mask, 'Age'].fillna(train[train.Title == 'Miss'].Age.mean())
train.loc[mrs_mask, 'Age'] = train.loc[mrs_mask, 'Age'].fillna(train[train.Title == 'Mrs'].Age.mean())
train.loc[master_mask, 'Age'] = train.loc[master_mask, 'Age'].fillna(train[train.Title == 'Master'].Age.mean())
train.loc[dr_mask, 'Age'] = train.loc[dr_mask, 'Age'].fillna(train[train.Title == 'Dr'].Age.mean())
print()
print(getTitle(train[train.Age.isnull()].Name).value_counts())
print(getTitle(test[test.Age.isnull()].Name).value_counts())

mr_mask = test['Title'] == 'Mr'
miss_mask = test['Title'] == 'Miss'
mrs_mask = test['Title'] == 'Mrs'
master_mask = test['Title'] == 'Master'
ms_mask = test['Title'] == 'Ms'
test.loc[mr_mask, 'Age'] = test.loc[mr_mask, 'Age'].fillna(test[test.Title == 'Mr'].Age.mean())
test.loc[miss_mask, 'Age'] = test.loc[miss_mask, 'Age'].fillna(test[test.Title == 'Miss'].Age.mean())
test.loc[mrs_mask, 'Age'] = test.loc[mrs_mask, 'Age'].fillna(test[test.Title == 'Mrs'].Age.mean())
test.loc[master_mask, 'Age'] = test.loc[master_mask, 'Age'].fillna(test[test.Title == 'Master'].Age.mean())
test.loc[ms_mask, 'Age'] = test.loc[ms_mask, 'Age'].fillna(test[test.Title == 'Ms'].Age.mean())
print(getTitle(test[test.Age.isnull()].Name).value_counts())
print(train.isna().sum())
print(test.isna().sum())
train.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, test]]
for df in [train, test]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)
[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, test]]
print(test)
[df.drop(columns=['Name'], inplace = True) for df in [train, test]]
[train, test] = [pd.get_dummies(data = df, columns = ['Title']) for df in [train, test]]
print(train.columns.values)
print(test.columns.values)
train.corr()
X = train[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)
print(y.value_counts())
y_default = pd.Series([0] * train['Survived'].shape[0], name = 'Survived')
print(y_default.value_counts())
print(confusion_matrix(y, y_default))
print()
print(accuracy_score(y, y_default))
#Initial accuracy
print("LinearSVC")
classifier = LinearSVC(dual = False)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("Logistic Regression")
classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("KNeighborsClassifier")
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("AdaBoostClassifier")
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print("BaggingClassifier")
classifier = BaggingClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
X_validation = test[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
y_valid = classifier.predict(X_validation)
validation_pId = test.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})
print(my_submission['Survived'].value_counts())
my_submission.to_csv('submission.csv', index = False)