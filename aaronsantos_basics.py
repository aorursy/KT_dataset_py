# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
len(train)
test = pd.read_csv('../input/test.csv')
test.head()
len(test)
train.isnull().sum()
def fill_median_age(df, sex):
    median_age = df[(df.Sex == sex)]['Age'].median()
    updated = df[(df.Sex == sex)].fillna({'Age': median_age})
    df.update(updated)
    return df
fill_median_age
def fill_embarked(df):
    embarked = df['Embarked'].mode().iloc[0]
    return df.fillna({'Embarked': embarked})
fill_embarked
def fill_fare(df):
    fare = df['Fare'].median()
    return df.fillna({'Fare': fare})
fill_fare
def drop_cabin(df):
    return df.drop(columns=['Cabin'])
drop_cabin
def clean(df):
    return (df
         .pipe(fill_median_age, sex = 'male')
         .pipe(fill_median_age, sex = 'female')
         .pipe(fill_embarked)
         .pipe(fill_fare)
         .pipe(drop_cabin)
    )
clean
clean_train = clean(train);
clean_train.head()
clean_train.isnull().sum()
clean_test = clean(test);
#X_test = clean_test.drop(columns = 'Survived')
#y_test = clean_test['Survived']
clean_test.head()
clean_test.isnull().sum()
sb.heatmap(clean_train.corr(), annot=True, fmt=".2f")
sb.pairplot(clean_train, hue = 'Pclass')
sb.pairplot(clean_train, hue = 'Sex')
sb.countplot(x = 'Survived', hue = 'Pclass', data = clean_train)
g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "Age")
g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "Parch")
g = sb.FacetGrid(clean_train, hue="Survived", size = 5)
g.map(sb.distplot, "SibSp")
from sklearn.model_selection import train_test_split
X = clean_train.drop(columns = 'Survived')
y = clean_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn_pandas import DataFrameMapper, cross_val_score
# Pass values through and binarize the Sex column
mapper = DataFrameMapper([
    ('Pclass', None),
    ('Age', None),
    ('Parch', None),
    ('SibSp', None),
    ('Sex', LabelBinarizer())
])
pipe = Pipeline([
    ('featurize', mapper),
    ('lm', DummyClassifier(strategy='most_frequent',random_state=0))])
clf = pipe.fit(X_train, y_train)
print("mean accuracy")
print("train: ", clf.score(X_train, y_train))
print("test: ", clf.score(X_test, y_test))
from sklearn.linear_model import LinearRegression

pipe = Pipeline([
    ('featurize', mapper),
    ('lm', LinearRegression())])
clf = pipe.fit(X_train, y_train)
print("train: ", clf.score(X_train, y_train))
print("test: ", clf.score(X_test, y_test))
import sklearn.tree
def class_age(age):
    if age <= 18:
        return 0
    elif age <= 25:
        return 1
    elif age <= 35:
        return 2
    elif age <= 40:
        return 3
    elif age <= 60:
        return 4
    else:
        return 5

def class_parch(parch):
    if parch < 3:
        return parch
    else:
        return 3

def class_sibsp(sibsp):
    if sibsp == 1:
        return 1
    else:
        return 0
    
def lift_to_array(func):
    return lambda X: np.vectorize(func)(X)
featurizer = DataFrameMapper([
        # Scaling features doesn't improve performace in this example, but it's good to get into the habit
        (['Pclass'], StandardScaler()),
        (['Fare'], StandardScaler()),
        (['Age'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('age_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_age))),
            ('age_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        (['Parch'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('parch_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_parch))),
            ('parch_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        (['SibSp'], Pipeline([
            #('age_scaler', StandardScaler()),
            ('sibsp_func', sklearn.preprocessing.FunctionTransformer(lift_to_array(class_sibsp))),
            ('sibsp_enc', sklearn.preprocessing.OneHotEncoder()),
        ])),
        ('Sex', LabelBinarizer())
    ])
pipe = Pipeline([
    ('featurize', featurizer),
    ('lm', sklearn.linear_model.LogisticRegression())
])
#np.round(cross_val_score(pipe, X=clean_train.copy(), y=clean_train['Survived'], cv=20, scoring='r2'), 2)
clf = pipe.fit(X_train, y_train)
print("train: ", clf.score(X_train, y_train))
print("test:  ", clf.score(X_test, y_test))
# Use the model to make predictions
y_predicted = clf.predict(X_test)

import scikitplot as skplt
import matplotlib.pyplot as plt

#skplt.metrics.plot_roc_curve(y_test, y_predicted)
skplt.metrics.plot_confusion_matrix(y_test, y_predicted)
plt.show()

print(sklearn.metrics.classification_report(y_test, y_predicted))
y_probas = clf.predict_proba(X_test)
skplt.metrics.plot_roc_curve(y_test, y_probas)
plt.show()
predicted = clf.predict(clean_test)
logistic_submission = pd.DataFrame(
    {'PassengerId': clean_test.PassengerId,
     'Survived': predicted}).astype('int32')
# you could use any filename. We choose submission here
logistic_submission.to_csv('submission.csv', index=False)
logistic_submission.head()
