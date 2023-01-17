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

        

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set_style('whitegrid')



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



from sklearn.metrics import mean_squared_error



import warnings

warnings.filterwarnings(action="ignore")



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv', index_col='PassengerId')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv', index_col='PassengerId')

df.head()
df.info()
df.describe()
df.describe(include=['O'])
# Understanding the target class is very important



sns.countplot('Survived', data=df)

100.0*df['Survived'].value_counts() / len(df)

df.corr()['Survived']
df[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()
df[['Sex', 'Survived']].groupby('Sex', as_index=False).mean()
df[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean().sort_values('Survived', ascending=False)
df[['Parch','Survived']].groupby('Parch', as_index=False).mean().sort_values('Survived', ascending=False)
g = sns.FacetGrid(df, col='Survived')

g.map(plt.hist, 'Age', bins=15)
g = sns.FacetGrid(df, col='Survived', row='Pclass')

g.map(plt.hist, 'Age')
df[['Embarked','Survived']].groupby('Embarked', as_index=False).mean().sort_values('Survived', ascending=False)
100.0*df['Embarked'].value_counts() / len(df)
df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')

test_df['Title'] = test_df.Name.str.extract('([A-Za-z]+)\.')
df.head()
df['Title'].value_counts()
pd.crosstab(df['Title'], df['Sex'])
replace_titles = ['Capt','Col','Countess','Don','Jonkheer','Lady','Major','Dr','Rev','Sir']
df['Title'] = df['Title'].replace(replace_titles, 'other')

df['Title'] = df['Title'].replace('Mlle', 'Miss')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')
df[['Title','Survived']].groupby('Title').mean().sort_values('Survived', ascending=False)
test_df['Title'] = test_df['Title'].replace(replace_titles, 'other')

test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')

test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')

test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
pd.crosstab(test_df['Title'], test_df['Sex'])
test_df['Title'] = test_df['Title'].replace('Dona', 'other')
df.isnull().sum().sort_values(ascending=False)
test_df.isnull().sum().sort_values(ascending=False)
print('Number of age entries missing for title Miss:', df[df['Title'] == 'Miss']['Age'].isnull().sum())

print('Number of age entries missing for title Mr:', df[df['Title'] == 'Mr']['Age'].isnull().sum())

print('Number of age entries missing for title Mrs:', df[df['Title'] == 'Mrs']['Age'].isnull().sum())

print('Number of age entries missing for title other:', df[df['Title'] == 'other']['Age'].isnull().sum())

print('Number of age entries missing for title Master:', df[df['Title'] == 'Master']['Age'].isnull().sum())
print('Mean age for title Miss:', df[df['Title'] == 'Miss']['Age'].mean())

print('Mean age for title Mr:', df[df['Title'] == 'Mr']['Age'].mean())

print('Mean age for title Mrs:', df[df['Title'] == 'Mrs']['Age'].mean())

print('Mean age for title other:', df[df['Title'] == 'other']['Age'].mean())

print('Mean age for title Master:', df[df['Title'] == 'Master']['Age'].mean())
df.loc[(df['Title']== 'Miss') & (df['Age'].isnull()), 'Age'] = 22

df.loc[(df['Title']== 'Mr') & (df['Age'].isnull()), 'Age'] = 32

df.loc[(df['Title']== 'Mrs') & (df['Age'].isnull()), 'Age'] = 36

df.loc[(df['Title']== 'other') & (df['Age'].isnull()), 'Age'] = 46

df.loc[(df['Title']== 'Master') & (df['Age'].isnull()), 'Age'] = 5
df.isnull().sum().sort_values(ascending=False)
# Repeating the steps for test set



print('Number of age entries missing for title Miss:', test_df[test_df['Title'] == 'Miss']['Age'].isnull().sum())

print('Number of age entries missing for title Mr:', test_df[test_df['Title'] == 'Mr']['Age'].isnull().sum())

print('Number of age entries missing for title Mrs:', test_df[test_df['Title'] == 'Mrs']['Age'].isnull().sum())

print('Number of age entries missing for title other:', test_df[test_df['Title'] == 'other']['Age'].isnull().sum())

print('Number of age entries missing for title Master:', test_df[test_df['Title'] == 'Master']['Age'].isnull().sum())
print('Mean age for title Miss:', test_df[test_df['Title'] == 'Miss']['Age'].mean())

print('Mean age for title Mr:', test_df[test_df['Title'] == 'Mr']['Age'].mean())

print('Mean age for title Mrs:', test_df[test_df['Title'] == 'Mrs']['Age'].mean())

print('Mean age for title other:', test_df[test_df['Title'] == 'other']['Age'].mean())

print('Mean age for title Master:', test_df[test_df['Title'] == 'Master']['Age'].mean())
test_df.loc[(test_df['Title']== 'Miss') & (test_df['Age'].isnull()), 'Age'] = 22

test_df.loc[(test_df['Title']== 'Mr') & (test_df['Age'].isnull()), 'Age'] = 32

test_df.loc[(test_df['Title']== 'Mrs') & (test_df['Age'].isnull()), 'Age'] = 39

test_df.loc[(test_df['Title']== 'other') & (test_df['Age'].isnull()), 'Age'] = 44

test_df.loc[(test_df['Title']== 'Master') & (test_df['Age'].isnull()), 'Age'] = 7
test_df.isnull().sum().sort_values(ascending=False)
df['Embarked'] = df['Embarked'].fillna('S')
test_df.loc[test_df['Fare'].isnull()]
# Finding out the mean Fare for Pclass=3



test_df[test_df['Pclass']==3]['Fare'].mean()
test_df['Fare'] = test_df['Fare'].fillna(12.46)
# Creating a new column for age groups



df['AgeGroup'] = pd.cut(df['Age'],5)
df[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False)
df.loc[df['Age'] <= 16, 'Age'] = 4

df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 3

df.loc[(df['Age'] >48) & (df['Age'] <= 64), 'Age'] = 2

df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

df.loc[(df['Age'] > 64), 'Age'] = 0
df = df.drop('AgeGroup', axis=1)
test_df.loc[test_df['Age'] <= 16, 'Age'] = 4

test_df.loc[(test_df['Age'] > 32) & (test_df['Age'] <= 48), 'Age'] = 3

test_df.loc[(test_df['Age'] >48) & (test_df['Age'] <= 64), 'Age'] = 2

test_df.loc[(test_df['Age'] > 16) & (test_df['Age'] <= 32), 'Age'] = 1

test_df.loc[(test_df['Age'] > 64), 'Age'] = 0
df[['Fare','Pclass']].groupby('Pclass', as_index=False).mean()
df['Fare'].min()
df['Fare'].max()
df['fareband'] = pd.cut(df['Fare'], 4)
df[['fareband', 'Survived']].groupby('fareband', as_index=False).mean().sort_values('Survived', ascending=False)
df.loc[(df['Fare'] >= 384), 'Fare'] = 3

df.loc[(df['Fare'] >= 256) & (df['Fare'] < 384), 'Fare'] = 2

df.loc[(df['Fare'] >=128) & (df['Fare'] < 256), 'Fare'] = 1

df.loc[df['Fare'] < 128, 'Fare'] = 0
df = df.drop('fareband', axis=1)
# Repeating the steps for the test set



test_df.loc[(test_df['Fare'] >= 384), 'Fare'] = 3

test_df.loc[(test_df['Fare'] >= 256) & (test_df['Fare'] < 384), 'Fare'] = 2

test_df.loc[(test_df['Fare'] >=128) & (test_df['Fare'] < 256), 'Fare'] = 1

test_df.loc[test_df['Fare'] < 128, 'Fare'] = 0
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
drop_cols = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin']
df = df.drop(drop_cols, axis=1)

test_df = test_df.drop(drop_cols, axis=1)
dummy_cols = ['Pclass','Sex', 'Age',  'Fare', 'Embarked', 'Title', 'FamilySize']

prefix_cats = ['pcl', 'sex', 'age', 'fare', 'emb', 'title', 'fsize']



df = pd.get_dummies(df, columns=dummy_cols, prefix=prefix_cats, drop_first=True)

test_df = pd.get_dummies(test_df, columns=dummy_cols, prefix=prefix_cats, drop_first=True)
X = df.drop('Survived', axis=1)

y = df['Survived']
X.head()
y.head()
# Creating an empty dataframe to add model predictions for comparison



pred_df = pd.DataFrame()
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
# initialize all the predictors and fit the training data



log_clf = LogisticRegression(random_state=42)

log_clf.fit(X, y)



sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X, y)



svc_clf = SVC(random_state=42)

svc_clf.fit(X, y)



tree_clf = DecisionTreeClassifier(random_state=42)

tree_clf.fit(X, y)



forest_clf = RandomForestClassifier(random_state=42)

forest_clf.fit(X, y)



extra_clf = ExtraTreesClassifier(random_state=42)

extra_clf.fit(X, y)



gb_clf = GradientBoostingClassifier(random_state=42)

gb_clf.fit(X, y)

# cross_val_predict and generate accuracy scores for all the predictors



log_preds = cross_val_predict(log_clf, X, y, cv=10)

log_acc = accuracy_score(y, log_preds)



sgd_preds = cross_val_predict(sgd_clf, X, y, cv=10)

sgd_acc = accuracy_score(y, sgd_preds)



svc_preds = cross_val_predict(svc_clf, X, y, cv=10)

svc_acc = accuracy_score(y, svc_preds)



tree_preds = cross_val_predict(tree_clf, X, y, cv=10)

tree_acc = accuracy_score(y, tree_preds)



forest_preds = cross_val_predict(forest_clf, X, y, cv=10)

forest_acc = accuracy_score(y, forest_preds)



extra_preds = cross_val_predict(extra_clf, X, y, cv=10)

extra_acc = accuracy_score(y, extra_preds)



gb_preds = cross_val_predict(gb_clf, X, y, cv=10)

gb_acc = accuracy_score(y, gb_preds)
print('log_clf', log_acc)

print('sgd_clf', sgd_acc)

print('svc_clf', svc_acc)

print('tree_clf', tree_acc)

print('forest_clf', forest_acc)

print('extra_clf', extra_acc)

print('gb_clf', gb_acc)
# Generating paramater grids for predictors



log_param = [

    {#'penalty':['l1', 'l2', 'elasticnet'],

    'C':[0.001, 0.01, 0.1, 1.0, 10.0]

    }

]



sgd_param = [

    {'alpha':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0],

    #'penalty':['l1', 'l2', 'elasticnet']

    }

]



svc_param = [

    {'C':[0.001, 0.01, 0.1, 1.0, 10.0],

    'gamma':[0.001, 0.01, 0.1, 1.0],

    'kernel':['rbf', 'sigmoid']}

]



tree_param = [

    {'max_depth':[2,4,8,12,16,20,30],

    'min_samples_split':[2,4,6,8,10],

    'min_samples_leaf':[2,4,6,8,10]

    }

]



forest_param = [

    {'max_depth':[2,4,8,12,16,20],

    'min_samples_split':[2,4,6,8,10],

    'min_samples_leaf':[2,4,6,8,10],

    'n_estimators':[100,200,300]}

]



extra_param = [

    {'max_depth':[2,4,8,12,16,20,30],

    'min_samples_split':[2,4,6,8,10],

    'min_samples_leaf':[2,4,6,8,10]}

]



gb_param = [

    {'max_depth':[2,8,16,20],

    'min_samples_split':[2,4,6,10],

    'min_samples_leaf':[2,4,6,10],

    'learning_rate':[0.01, 0.05, 0.1],

    'n_estimators':[100,200,300],

    'subsample':[0.5, 0.8, 1.0]}

]

log_grid = GridSearchCV(log_clf, log_param, cv=5)

log_grid.fit(X, y)

log_best = log_grid.best_estimator_
sgd_grid = GridSearchCV(sgd_clf, sgd_param, cv=5)

sgd_grid.fit(X, y)
sgd_best = sgd_grid.best_estimator_
svc_grid = GridSearchCV(svc_clf, svc_param, cv=5)

svc_grid.fit(X, y)
svc_best = svc_grid.best_estimator_
tree_grid = GridSearchCV(tree_clf, tree_param, cv=5)

tree_grid.fit(X, y)
tree_best = tree_grid.best_estimator_
forest_grid = GridSearchCV(forest_clf, forest_param, cv=5, verbose=1, n_jobs=-1)

forest_grid.fit(X, y)
forest_best = forest_grid.best_estimator_
extra_grid = GridSearchCV(extra_clf, extra_param, cv=5, verbose=1, n_jobs=-1)

extra_grid.fit(X, y)
extra_best = extra_grid.best_estimator_
gb_grid = GridSearchCV(gb_clf, gb_param, cv=5, verbose=1, n_jobs=-1)

gb_grid.fit(X, y)
gb_best = gb_grid.best_estimator_
log_best.fit(X, y)
sgd_best.fit(X, y)
svc_best.fit(X, y)
tree_best.fit(X, y)
forest_best.fit(X, y)
extra_best.fit(X, y)
gb_best.fit(X, y)
log_best_preds = cross_val_predict(log_best, X, y, cv=10)

log_best_acc = accuracy_score(y, log_best_preds)



sgd_best_preds = cross_val_predict(sgd_best, X, y, cv=10)

sgd_best_acc = accuracy_score(y, sgd_best_preds)



svc_best_preds = cross_val_predict(svc_best, X, y, cv=10)

svc_best_acc = accuracy_score(y, svc_best_preds)



tree_best_preds = cross_val_predict(tree_best, X, y, cv=10)

tree_best_acc = accuracy_score(y, tree_best_preds)



forest_best_preds = cross_val_predict(forest_best, X, y, cv=10)

forest_best_acc = accuracy_score(y, forest_best_preds)



extra_best_preds = cross_val_predict(extra_best, X, y, cv=10)

extra_best_acc = accuracy_score(y, extra_best_preds)



gb_best_preds = cross_val_predict(gb_best, X, y, cv=10)

gb_best_acc = accuracy_score(y, gb_best_preds)
pred_df = pred_df.append({'b.Best Estimtor Accuracy': log_best_acc, 'b.Accuracy': log_acc, 'a.Model':'log_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': sgd_best_acc, 'b.Accuracy': sgd_acc, 'a.Model':'sgd_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': svc_best_acc, 'b.Accuracy': svc_acc, 'a.Model':'svc_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': tree_best_acc, 'b.Accuracy': tree_acc, 'a.Model':'tree_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': forest_best_acc, 'b.Accuracy': forest_acc, 'a.Model':'forest_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': extra_best_acc, 'b.Accuracy': extra_acc, 'a.Model':'extra_clf'}, ignore_index=True)

pred_df = pred_df.append({'b.Best Estimtor Accuracy': gb_best_acc, 'b.Accuracy': gb_acc, 'a.Model':'gb_clf'}, ignore_index=True)
pred_df
svc_test_preds = svc_best.predict(test_df)
gb_test_preds = gb_best.predict(test_df)
forest_test_preds = forest_best.predict(test_df)
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

submission['Survived'] = svc_test_preds

submission.to_csv('svc_final_submission.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

submission['Survived'] = gb_test_preds

submission.to_csv('gb_final_submission.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv', index_col='PassengerId')

submission['Survived'] = forest_test_preds

submission.to_csv('forest_final_submission.csv')
#X = pd.get_dummies(new_train[features])

#X_test = pd.get_dummies(new_test[features])

# X = new_train.drop("Survived", axis=1)

# X_test = new_test



#model = XGBClassifier(max_depth=3, n_estimators=1000, learning_rate=0.05)

#model = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=1)

#model = XGBClassifier(max_depth=5, n_estimators=1000, learning_rate=0.05)

#model = XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05)

#model.fit(X, y)

#predictions = model.predict(X_test)



#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

#output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")
