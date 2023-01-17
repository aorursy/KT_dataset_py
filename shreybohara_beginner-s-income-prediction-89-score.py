import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')



from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import roc_auc_score



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/adult-census-income/adult.csv')
df.head()
df.info()
df.describe()
df.shape
df.select_dtypes(exclude=np.number).columns
df.select_dtypes(include=np.number).columns
# Checking for null values, if any



df.isnull().sum()
# Checking for class imbalance



df['income'].value_counts()
# Converting the same into percentage, for better understanding



df['income'].value_counts(normalize=True)*100
plt.figure(figsize=(15,8))

sns.countplot(df['workclass'], hue=df['income']);
plt.figure(figsize=(15,8))

ax = sns.countplot(df['education'], hue=df['income']);



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(df['marital.status'], hue=df['income']);
plt.figure(figsize=(15,8))

ax = sns.countplot(df['occupation'], hue=df['income']);



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,8))

sns.countplot(df['relationship'], hue=df['income']);
plt.figure(figsize=(15,8))

sns.countplot(df['race'], hue=df['income']);
sns.countplot(df['sex'], hue=df['income']);
plt.figure(figsize=(15,8))

ax = sns.countplot(df['native.country']);



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,8))

sns.distplot(df['age']);
plt.figure(figsize=(15,8))

sns.distplot(df['fnlwgt']);
plt.figure(figsize=(15,8))

sns.distplot(df['education.num']);
plt.figure(figsize=(15,8))

sns.distplot(df['capital.gain'], kde=False);
plt.figure(figsize=(15,8))

sns.distplot(df['capital.loss'], kde=False);
plt.figure(figsize=(15,8))

sns.distplot(df['hours.per.week']);
# Checking correlation



sns.heatmap(df.corr(), annot=True, cmap='viridis');
# Number of '?' in the dataset



for col in df.columns:

    print(col,':', df[df[col] == '?'][col].count())
for cols in df.select_dtypes(exclude=np.number).columns:

    df[cols] = df[cols].str.replace('?', 'Unknown')
# Unique values in each categorical feature



for cols in df.select_dtypes(exclude=np.number).columns:

    print(cols, ':', df[cols].unique(), end='\n\n')
# Checking for correlation between columns 'education' and 'education-num'



pd.crosstab(df['education.num'],df['education'])
df['native.country'].value_counts(normalize=True)*100
df.drop(['fnlwgt', 'capital.gain', 'capital.loss', 'native.country', 'education'], axis=1, inplace=True)
# Dropping rows with hours.per.week = 99



df.drop(df[df['hours.per.week'] == 99].index, inplace=True)
# Converting values in target column to numbers



df['income'] = df['income'].map({'<=50K':0, '>50K':1})
# Encoding categorical features



categorical_columns = df.select_dtypes(exclude=np.number).columns

new_df = pd.get_dummies(data=df, prefix=categorical_columns, drop_first=True)
new_df.shape
pd.set_option('max_columns', 50)

new_df.head()
X = new_df.drop('income', axis=1)

y = new_df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Hyperparameter tuning of Logistic Regression



param_grid = {'penalty':['l1', 'l2', 'elasticnet'], 'C':[0.001, 0.01, 0.1, 1, 10, 100],

             'solver':['lbfgs', 'liblinear'], 'l1_ratio':[0.001, 0.01, 0.1]}



grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, verbose=3)



grid.fit(X, y)
grid.best_params_
grid.best_score_
log_reg = LogisticRegression(C=1, l1_ratio=0.001, solver='lbfgs', penalty='l2')
# Hyperparameter tuning of Random Forest



param_grid = {'criterion':['gini', 'entropy'], 'max_depth':[2, 4, 5, 7, 9, 10], 'n_estimators':[100, 200, 300, 400, 500]}



grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, verbose=3)



grid.fit(X, y)
grid.best_params_
grid.best_score_
rfc = RandomForestClassifier(max_depth=10, n_estimators=100, criterion='gini')
# Hyperparameter tuning of XGBoost



param_grid = {'max_depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'min_child_weight':[2, 4, 5, 6, 7]}



grid = GridSearchCV(XGBClassifier(), param_grid=param_grid, verbose=3)



grid.fit(X, y)
grid.best_params_
grid.best_score_
xgb = XGBClassifier(learning_rate=0.2, max_depth=4, min_child_weight=2)
# Hyperparameter tuning of CatBoost



param_grid = {'depth':[2, 4, 5, 7, 9, 10], 'learning_rate':[0.001, 0.01, 0.1, 0.2, 0.3], 'iterations':[30, 50, 100]}



grid = GridSearchCV(CatBoostClassifier(), param_grid, verbose=3)



grid.fit(X, y)
grid.best_params_
grid.best_score_
cb = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.1, verbose=False)
classifiers = [log_reg, rfc, xgb, cb]



folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)



scores_dict = {}



for train_index, valid_index in folds.split(X_train, y_train):

    # Need to use iloc as it provides integer-location based indexing, regardless of index values.

    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]

    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

    

    for classifier in classifiers:

        name = classifier.__class__.__name__

        classifier.fit(X_train_fold, y_train_fold)

        training_predictions = classifier.predict_proba(X_valid_fold)

        # roc_auc_score should be calculated on probabilities, hence using predict_proba

        

        scores = roc_auc_score(y_valid_fold, training_predictions[:, 1])

        if name in scores_dict:

            scores_dict[name] += scores

        else:

            scores_dict[name] = scores



# Taking average of the scores

for classifier in scores_dict:

    scores_dict[classifier] = scores_dict[classifier]/folds.n_splits
scores_dict
final_predictions = xgb.predict_proba(X_test)



print(roc_auc_score(y_test, final_predictions[:, 1]))