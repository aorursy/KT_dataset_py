# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import glob
import os
import sys
from IPython.core.display import HTML

HTML("""
<style>
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style>
""");
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.options.display.max_columns = 100
random_state=42
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv('../input/titanic/train.csv')
test_df=pd.read_csv('../input/titanic/test.csv')
gender_df=pd.read_csv('../input/titanic/gender_submission.csv')
os.chdir('/kaggle/working/')
for i in train_df.columns:
    print(f"column name - {i} and their type is {type(train_df[i][0])}")
# check sample dataset what it looks like.
train_df.head(2)
train_df.tail(2)
# Info of all the Features of Dataset -their Types, Count, Null r Not Null.
train_df.info()
train_df.describe(include=['O'])
train_df.describe(exclude=['O'])
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=30)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', bins=30)
grid.add_legend();
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', ci=None)
grid.add_legend()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', bins=20)
grid.add_legend()
# train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
# train_df['Died']=1-train_df['Survived']
train_df.corr()
complete_df=pd.concat([train_df,test_df])

assert complete_df.shape[0]==train_df.shape[0]+test_df.shape[0]

complete_df.dtypes
complete_df.drop(['Survived'],1,inplace=True)

complete_df.reset_index(inplace=True)

complete_df.drop(['index','PassengerId'],1,inplace=True)

complete_df.shape
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
complete_df['Title'] = complete_df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

complete_df['Title']=complete_df.Title.map(Title_Dictionary)

complete_df
print(complete_df.iloc[:891].Age.isnull().sum(),complete_df.iloc[891:].Age.isnull().sum())

median_group_age=complete_df.iloc[:891].groupby(['Sex','Pclass','Title']).median().reset_index()[['Sex', 'Pclass', 'Title', 'Age']]

median_group_age.head(5)
def fill_age(row):
    return median_group_age[(median_group_age['Sex']==row['Sex']) & 
                            (median_group_age['Title']==row['Title']) & 
                            (median_group_age['Pclass']==row['Pclass'])]['Age'].values[0]
complete_df['Age'] = complete_df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)

print(complete_df['Age'].isnull().sum())

# complete_df
complete_df.drop('Name',axis=1,inplace=True)
title_dummies=pd.get_dummies(complete_df['Title'],prefix='Title')
complete_df = pd.concat([complete_df, title_dummies], axis=1)
complete_df.drop(['Title'],axis=1,inplace=True)
complete_df.shape
complete_df.Fare.fillna(complete_df.iloc[:891].Fare.mean(), inplace=True)
complete_df.Embarked.fillna('S', inplace=True)
embarked_dummies = pd.get_dummies(complete_df['Embarked'], prefix='Embarked')
complete_df=pd.concat([complete_df,embarked_dummies],axis=1)
complete_df.drop(['Embarked'],axis=1,inplace=True)
train_cabin, test_cabin = set(), set()
for c in complete_df.iloc[:891]['Cabin']:
    try:
        train_cabin.add(c[0])
    except:
        train_cabin.add('U')
for c in complete_df.iloc[891:]['Cabin']:
    try:
        test_cabin.add(c[0])
    except:
        test_cabin.add('U')
# print(train_cabin)
# print(test_cabin)
# print(complete_df['Cabin'].value_counts())
complete_df.Cabin.fillna('U', inplace=True)
complete_df['Cabin'] = complete_df['Cabin'].map(lambda c: c[0])
complete_df['Cabin'].value_counts()
cabin_dummies = pd.get_dummies(complete_df['Cabin'], prefix='Cabin') 
complete_df=pd.concat([complete_df,cabin_dummies],axis=1)
complete_df.drop('Cabin', axis=1, inplace=True)
complete_df['Sex']=complete_df.Sex.map({'male':1,'female':0})
pclass_dummies = pd.get_dummies(complete_df['Pclass'], prefix="Pclass")
complete_df=pd.concat([complete_df,pclass_dummies],axis=1)
complete_df.drop('Pclass',axis=1,inplace=True)
def cleanTicket(ticket):
    ticket=str(ticket)
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
complete_df['Ticket'] = complete_df['Ticket'].map(cleanTicket)
tickets_dummies = pd.get_dummies(complete_df['Ticket'], prefix='Ticket')
complete_df = pd.concat([complete_df, tickets_dummies], axis=1)
complete_df.drop('Ticket', inplace=True, axis=1)
complete_df
complete_df['FamilySize'] = complete_df['Parch'] + complete_df['SibSp'] + 1
complete_df['Singleton'] = complete_df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
complete_df['SmallFamily'] = complete_df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
complete_df['LargeFamily'] = complete_df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
# complete_df.drop(columns=['Died'],axis=1,inplace=True)
complete_df.shape

complete_df.to_csv('/kaggle/working/complete_titanic_data_20200525.csv')
complete_df.head(10)
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

y_train=train_df['Survived']
X_train=complete_df.iloc[:891]
X_test=complete_df.iloc[891:]

plt.figure(figsize=(30,30))
sns.heatmap(X_train.corr(), cmap= 'coolwarm')
plt.show()

clf = RandomForestClassifier(n_estimators=500, max_features='sqrt')
clf = clf.fit(X_train, y_train)

features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))
y_train=train_df['Survived']
X_train=complete_df.iloc[:891]
X_test=complete_df.iloc[891:]
plt.figure(figsize=(30,30))
sns.heatmap(X_train.corr(), cmap= 'coolwarm')
plt.show()
clf = RandomForestClassifier(n_estimators=500, max_features='sqrt')
clf = clf.fit(X_train, y_train)
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

features.plot(kind='barh', figsize=(25, 25))
features.sort_values(by=['importance'], ascending=False, inplace=True)

features
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(X_train)
print(train_reduced.shape)
# (891L, 14L)
test_reduced = model.transform(X_test)
print(test_reduced.shape)
logreg = LogisticRegression()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()
dt=DecisionTreeClassifier()
xgb=XGBClassifier()
models = [logreg,rf, gboost,dt,xgb]

for model in models:
    print('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=y_train, scoring='accuracy')
    print('CV score = {0}'.format(score))
    print('****')
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 100, num = 10)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 5, 
                               verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(train_reduced, y_train)
"""{'n_estimators': 2000,
 'min_samples_split': 2,
 'min_samples_leaf': 2,
 'max_features': 'auto',
 'max_depth': 90,
 'bootstrap': True}"""
rf_random.best_params_
parameters = {'bootstrap': True, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 90}
# paraeters ={'n_estimators': 2000,
#  'min_samples_split': 2,
#  'min_samples_leaf': 2,
#  'max_features': 'auto',
#  'max_depth': 90,
#  'bootstrap': True}
model = RandomForestClassifier(**parameters)
model = model.fit(train_reduced,y_train)
output = model.predict(test_reduced).astype(int)
df_output = pd.DataFrame()
aux=pd.read_csv("../input/titanic/test.csv")
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('/kaggle/working/RandomFor_pred.csv', index=False)
