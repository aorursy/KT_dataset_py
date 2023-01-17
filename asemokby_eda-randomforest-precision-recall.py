# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train, test = pd.read_csv('../input/titanic/train.csv'), pd.read_csv('../input/titanic/test.csv')
train.head()
train.info()
train['Sex'].value_counts()
train.describe()
train['Pclass'].value_counts()
train.hist(bins=50, figsize=(15, 10))
train_c = train.copy()
def cat_survival_rate(column_name):

    """

    Counting the people survived in each class. And calculating the survial ratio for each. 

    """

    cat_survived = train_c.groupby(column_name).agg({'PassengerId':'count', 'Survived':'sum'}

                                                   ).reset_index()

    cat_survived['survival_rate'] = cat_survived.Survived/cat_survived.PassengerId

    return cat_survived.rename(columns={'PassengerId':'PassengerCount'})
pclass_survival_ratio = cat_survival_rate('Pclass')

pclass_survival_ratio
def plot_cat_survived(df, colum_name):

    fig = plt.figure(figsize=(8, 5))

    plt.bar(df[colum_name]-.2, df.PassengerCount, width=.3,label='Passengers Count')

    plt.bar(df[colum_name]+.1, df.Survived, width=.3, label='Survived')

    plt.legend()

    plt.show()

plot_cat_survived(pclass_survival_ratio, 'Pclass')
def plot_cat_survival_ratio(df, column_name):

    f, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df[column_name], df.survival_rate,label='Survial Ratio', marker='s')

#     ax.set_xticks([1, 2, 3])

#     ax.set_xticklabels(['class 1', 'class 2', 'class 3'])

    plt.legend()

    plt.show()

plot_cat_survival_ratio(pclass_survival_ratio, 'Pclass')
parch_survival_ratio = cat_survival_rate('Parch')

parch_survival_ratio
plot_cat_survival_ratio(parch_survival_ratio, 'Parch')
sibsp_survival_ratio = cat_survival_rate('SibSp')

sibsp_survival_ratio
plot_cat_survival_ratio(sibsp_survival_ratio, 'SibSp')
train_c.groupby('Pclass').sum()
corr_matrix = train_c.corr()
corr_matrix
corr_matrix['Survived'].sort_values(ascending=False)
train_c['Parch_SibSp'] = train_c['SibSp'] + train_c['Parch']

train_c['age_parch'] = train_c['Parch']/train_c['Age']

train_c['age_Sibsp'] = train_c['Age']*train_c['SibSp']

corr_matrix = train_c.corr()

corr_matrix['Survived'].sort_values(ascending=False)
train_c = train.drop('Survived', axis=1)

train_c_labels = train['Survived'].copy()
train_c.info()
train_c.Age.hist()

plt.show()
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
num_attrs = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

imputer.fit(train_c[num_attrs])  

train_num = imputer.transform(train_c[num_attrs])
# converting the output back to a dataframe. 

train_num = pd.DataFrame(train_num, columns=train_c[num_attrs].columns, index=train_c[num_attrs].index)
train_num.info()
train_c.drop('Cabin', axis=1)
imputer_cat = SimpleImputer(strategy='most_frequent')

cat_attrs = ['Sex', 'Embarked']

train_cat = imputer_cat.fit_transform(train_c[cat_attrs])

train_cat = pd.DataFrame(train_cat, columns=train_c[cat_attrs].columns, index=train_c[cat_attrs].index)

train_cat.info()
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()

cat_ecnoded = cat_encoder.fit_transform(train_cat)

cat_ecnoded.toarray()
train_c.head()
from sklearn.base import BaseEstimator, TransformerMixin



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, idx):

        self.idx = idx

    def fit(self, X, y=None):

        return self

    def transform(self, X):        

        parch_SibSp = X[:,self.idx[1]]*X[:, self.idx[2]]

        age_parch = X[:, self.idx[2]]/X[:, self.idx[0]]

        age_Sibsp = X[:, self.idx[2]]*X[:, self.idx[1]]

        return np.c_[X, parch_SibSp, age_parch, age_Sibsp]

        

attr_adder = CombinedAttributesAdder(idx=[4, 5, 6])

train_extra_attrs = attr_adder.transform(train_c.values)
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='mean')),

    ('attrs_adder', CombinedAttributesAdder(idx=[0, 1, 2])),

    ('std_scaler', StandardScaler()),

])



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('encoder', OneHotEncoder())

])
from sklearn.compose import ColumnTransformer

num_attrs = ['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']

cat_attrs = ['Sex', 'Embarked']



full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attrs),

    ('cat', cat_pipeline, cat_attrs)

])



train_prepared = full_pipeline.fit_transform(train_c)

test_prepared = full_pipeline.fit_transform(test)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier()
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, train_prepared, train_c_labels, cv=3, scoring='accuracy')
train['Survived'].value_counts() #549/(342 + 549) ~= 61
from sklearn.model_selection import cross_val_predict

preds = cross_val_predict(sgd_clf, train_prepared, train_c_labels, cv=3)
from sklearn.metrics import confusion_matrix

confusion_matrix(train_c_labels, preds)
from sklearn.metrics import precision_score, recall_score

precision_score(train_c_labels, preds)
recall_score(train_c_labels, preds)
from sklearn.metrics import f1_score

f1_score(train_c_labels, preds)
scores = cross_val_predict(sgd_clf, train_prepared, train_c_labels, cv=3, method='decision_function')
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(train_c_labels, scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision") 

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.legend()

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
threshold_80_precision = thresholds[np.argmax([precisions >= .80])]

preds_80_precision = (scores >= threshold_80_precision)
precision_score(train_c_labels, preds_80_precision)
recall_score(train_c_labels, preds_80_precision)
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
preds = cross_val_predict(forest_clf, train_prepared, train_c_labels, cv=3)

precision_score(train_c_labels, preds)
recall_score(train_c_labels, preds)
f1_score(train_c_labels, preds)
from sklearn.linear_model import LogisticRegression

preds = cross_val_predict(LogisticRegression(), train_prepared, train_c_labels, cv=3)

precision_score(train_c_labels, preds)
recall_score(train_c_labels, preds)
f1_score(train_c_labels, preds)
from sklearn.model_selection import GridSearchCV



param_grid={"C":np.logspace(-3,3,7), "penalty":["l2"]}

logistic_clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, verbose=0)

logistic_clf_grid = logistic_clf.fit(train_prepared, train_c_labels)
logistic_clf_grid.best_params_
lg_best = LogisticRegression(C = 0.1, penalty='l2')

preds = cross_val_predict(lg_best, train_prepared, train_c_labels, cv=3)

f1_score(train_c_labels, preds)
logistic_final_preds = logistic_clf_grid.predict(test_prepared)
submit = pd.read_csv('../input/titanic/gender_submission.csv')

submit['Survived'] = logistic_final_preds

submit.to_csv('logistic_submission.csv', index=False)
param_grid = { 

    'n_estimators': [80, 100, 120, 140, 150],

    'max_features': ['auto', 'sqrt', 'log2']

}

forest_clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, verbose=0)

forest_clf_grid = forest_clf.fit(train_prepared, train_c_labels)
forest_clf_grid.best_params_
forest_best = RandomForestClassifier(max_features='sqrt', n_estimators=140)

preds = cross_val_predict(forest_best, train_prepared, train_c_labels, cv=3)

f1_score(train_c_labels, preds)
forest_final_preds = forest_clf_grid.predict(test_prepared)
submit = pd.read_csv('../input/titanic/gender_submission.csv')

submit['Survived'] = forest_final_preds

submit.to_csv('forest_submission.csv', index=False)