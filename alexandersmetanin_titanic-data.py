# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline 

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
gender_submission_df = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")
train_df = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")
train_df.head()
print('Total number of passangers in data set:', train_df.shape[0] + test_df.shape[0])
print('Passangers in train:', train_df.shape[0])
print('Passangers in test:', test_df.shape[0])
# checking data types
train_df.dtypes
# Looking at useful stats (mean, std, etc) for numeric columns to understand data
train_df.describe()
# Looking at non-numeric columns
train_df.describe(exclude=[np.number])
# Checking missing data and null values
train_missing_data = train_df.isnull().sum().sort_values(ascending=False)
train_missing_data
# Check survival rates in different categories
pd.pivot_table(train_df, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])

sns.barplot(x='Sex', y='Survived', data=train_df)
# Survival ratio in different class
sns.barplot(x='Pclass', y='Survived', data=train_df)
sns.barplot(x='Parch', y='Survived', data=train_df)
# Exploring age
grid = sns.FacetGrid(train_df, col = 'Sex', hue = 'Survived')
grid.map(plt.hist, 'Age', alpha = .75)
grid.add_legend()
# Looking at Fare importance
sns.scatterplot(x="Age", y="Fare", hue="Survived", legend="full", data=train_df)
# Look at correlations
numerical_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_df = train_df[numerical_cols].corr()
print(corr_df)
sns.heatmap(corr_df)
def fill_age(df):
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
    return df
# Fill age for train and test
train_df = fill_age(train_df)
test_df = fill_age(test_df)

# Fill embarked with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode())
train_df.loc[~train_df['Embarked'].isin(['S', 'Q', 'C']), 'Embarked'] = 'S'

# Fill Fare with mean
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())

# Cabin columns has too many missing values
for df in (train_df, test_df):
    df['Cabin'] = df['Cabin'].fillna('M')
# Extract Title (Miss, Mr, etc) from name
for df in (train_df, test_df):
    df['Title'] = df['Name'].str.split(' ', expand=True)[1].str[:-1]
    common_titles = df['Title'].value_counts() < 20
    df['Title'] = df['Title'].apply(lambda x: 'Other' if common_titles.loc[x] else x)
train_df['Title'].value_counts()
pd.qcut(train_df['Age'], 5).value_counts()
# split into groups by age
def get_age_group(age):
    if age <= 20: return 0
    if age <= 25: return 1
    if age <= 30: return 2
    if age <= 40: return 3
    return 4

train_df['AgeGroup'] = train_df['Age'].apply(get_age_group)
test_df['AgeGroup'] = test_df['Age'].apply(get_age_group)
train_df['AgeGroup']
pd.qcut(train_df['Fare'], 5).value_counts()
# split into groups by age
def get_fare_group(age):
    if age <= 7.85: return 0
    if age <= 10.5: return 1
    if age <= 21.7: return 2
    if age <= 40: return 3
    return 4

train_df['FareGroup'] = train_df['Fare'].apply(get_age_group)
test_df['FareGroup'] = test_df['Fare'].apply(get_age_group)
# Using log transformation
train_df['LogFare'] = np.log(train_df['Fare'].values)
test_df['LogFare'] = np.log(test_df['Fare'].values)
train_df['SexNum'] = (train_df['Sex'] == 'male').astype(int)
test_df['SexNum'] = (test_df['Sex'] == 'male').astype(int)
for df in (train_df, test_df):
    df['Deck'] = df['Cabin'].str[0]

test_df['Deck'].value_counts()
train_df.groupby(['Deck', 'Sex']).agg({'Survived': 'mean'})
# Decks A, B, D, F are simillar and will be merged, decks G, M, T as well
for df in (train_df, test_df):
    df['Deck'] = df['Deck'].replace(['A', 'B', 'D', 'F'], 'ABDF')
    df['Deck'] = df['Deck'].replace(['C', 'E'], 'CE')
    df['Deck'] = df['Deck'].replace(['M', 'G', 'T'], 'GMT')

train_df['Deck'].value_counts()
# Adding numeric ticket category
for df in (train_df, test_df):
    df['NumericTicket'] = df['Ticket'].str.isnumeric().astype(int)
train_df
fig, axs = plt.subplots(nrows=2, figsize=(25, 20))

sns.heatmap(train_df.corr(), ax=axs[0], annot=True, square=True, cmap="YlGnBu") 
sns.heatmap(test_df.corr(), ax=axs[1], annot=True, square=True)

    
axs[0].set_title('Correlations for train_df')
axs[1].set_title('Correlations for test_df')

plt.show()
y = train_df['Survived'].values
reduced_train_df = train_df.drop(['Survived', 'Name', 'Age', 'Ticket', 'Fare', 'Sex', 'Cabin'], axis=1)
reduced_train_df
# columns to encode: Embarked, Title, Deck
encode_cols = ['Embarked', 'Title', 'Deck']
df_list = []
for col in encode_cols:
    df_list.append(pd.get_dummies(reduced_train_df[col], prefix=col, drop_first=True))
df_list.append(reduced_train_df.drop(encode_cols, axis=1))
final_train_df = pd.concat(df_list, axis=1)
final_train_df
# same for test
reduced_test_df = test_df.drop(['Name', 'Age', 'Ticket', 'Fare', 'Sex', 'Cabin', 'FareGroup'], axis=1)
encode_cols = ['Embarked', 'Title', 'Deck']
df_list = []
for col in encode_cols:
    df_list.append(pd.get_dummies(reduced_test_df[col], prefix=col, drop_first=True))
df_list.append(reduced_test_df.drop(encode_cols, axis=1))
final_test_df = pd.concat(df_list, axis=1)
final_test_df
# Compare train and test df
print(list(final_train_df))
print(list(final_test_df))
# Title_Miss col is missing in test
final_test_df['Title_Miss'] = 0
final_test_df = final_test_df[list(final_train_df)]
final_test_df
print(list(final_train_df))
print(list(final_test_df))
x_train = final_train_df.values
y_train = y
x_test = final_test_df.values
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
# Using naive bayes classifier as baseline
nb_clf = GaussianNB()
nb_cv = cross_val_score(nb_clf, x_train, y_train, cv=5).mean()
print('CV score with Naive Bayes:', nb_cv)
lr_clf = LogisticRegression(max_iter = 2000)
lr_cv = cross_val_score(lr_clf, x_train, y_train, cv=5).mean()
print('CV score with Logistic Regression:', lr_cv)
dt_clf = DecisionTreeClassifier()
dt_cv = cross_val_score(dt_clf, x_train, y_train, cv=5).mean()
print('CV score with Decision Tree:', dt_cv)
knn_clf = KNeighborsClassifier()
knn_cv = cross_val_score(knn_clf, x_train, y_train, cv=5).mean()
print('CV score with KNN:', knn_cv)
rf_clf = RandomForestClassifier()
rf_cv = cross_val_score(rf_clf, x_train, y_train, cv=5).mean()
print('CV score with Random Forest:', rf_cv)
svc_clf = SVC(probability=True)
svc_cv = cross_val_score(svc_clf, x_train, y_train, cv=5).mean()
print('CV score with SupportVectorClassifier:', svc_cv)
xgb_clf = XGBClassifier()
xgb_cv = cross_val_score(xgb_clf, x_train, y_train, cv=5).mean()
print('CV score with XGB:', xgb_cv)
cat_clf = CatBoostClassifier(logging_level='Silent')
cat_cv = cross_val_score(cat_clf, x_train, y_train, cv=5).mean()
print('CV score with CatBoost:', cat_cv)
comp_df = pd.DataFrame({
    'Model': ['NaiveBayes', 'LogisticRegression', 'DecisionTree', 'KNearestNegihbours', 'RandomForest', 
              'SupportVectorClassifier', 'XGBoost', 'CatBoost'],
    'CV Score': [nb_cv, lr_cv, dt_cv, knn_cv, rf_cv, svc_cv, xgb_cv, cat_cv]
})
ax = sns.barplot(x="Model", y="CV Score", data=comp_df, palette="Set2")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
def classifier_performance(clf, model):
    print(model)
    print('Best Score: ' + str(clf.best_score_))
    print('Best Parameters: ' + str(clf.best_params_))
# lr_clf = LogisticRegression()
# param_grid = {'max_iter' : [3000],
#               'penalty' : ['l1', 'l2', 'elasticnet'],
#               'C' : np.logspace(-4, 4, 20),
#               'solver' : ['liblinear']}

# gscv_lr = GridSearchCV(lr_clf, param_grid=param_grid, cv=5, verbose=True)
# best_lr_clf = gscv_lr.fit(x_train, y_train)
# classifier_performance(best_lr_clf,'Logistic Regression')
# svc_clf = SVC(probability = True)
# param_grid = [{'kernel': ['rbf'], 'gamma': [.1,.5,1,2,5,10], 'C': [.1, 1, 10, 100, 1000]},
#               {'kernel': ['linear'], 'C': [.1, 1, 10, 100, 1000]},
#               {'kernel': ['poly'], 'degree': [2,3,4,5], 'C': [.1, 1, 10, 100, 1000]}]
# gscv_svc = GridSearchCV(svc_clf, param_grid=param_grid, cv=5, verbose=True)
# best_svc_clf = gscv_svc.fit(x_train, y_train)
# classifier_performance(best_svc_clf,'SVC')
# cat_clf = CatBoostClassifier(logging_level='Silent')
# grid = {'learning_rate': [0.03, 0.1],
#         'depth': [4, 6, 10],
#         'l2_leaf_reg': [1, 3, 5, 7, 9]}

# grid_search_result = cat_clf.grid_search(grid, cv=5,
#                                          X=x_train, 
#                                          y=y_train)
# best_params = grid_search_result['params']
# cat_clf = CatBoostClassifier(logging_level='Silent', **best_params)
# cat_cv = cross_val_score(cat_clf, x_train, y_train, cv=5).mean()
# print('CV score with tuned CatBoost:', cat_cv)
# cat_clf.fit(x_train, y_train)
# cat_pred = cat_clf.predict(x_test)
rf_cls = RandomForestClassifier()
rf_cls.fit(x_train, y_train)
y_test = rf_cls.predict(x_test)
df_submit = pd.DataFrame(columns=['PassengerId', 'Survived'])
df_submit['PassengerId'] = test_df.index
df_submit['Survived'] = cat_pred
df_submit.to_csv('submission_02_cat.csv', header=True, index=False)
