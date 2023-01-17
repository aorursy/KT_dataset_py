import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train = pd.read_csv('../input/titanic/train.csv')
train['Survived'].value_counts().plot.bar()

plt.title('Counts of Survived')

plt.figtext(0.90, 0.01, '0 = No, 1 = Yes', horizontalalignment='right')

plt.xticks(rotation=360);
print('Count:')

print(train['Survived'].value_counts())

print('\n')

print('Percent:')

print(train['Survived'].value_counts() / len(train) * 100)

print('\n')

print('0 = No, 1 = Yes')
X_train = train.drop(['Survived'], axis=1)

y_train = train['Survived'].copy()
# For numerical attributes

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')), # impute missing values with median

    ('minmax_scaler', MinMaxScaler()),             # scale features

])
# For categorical attributes

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')), # impute missing values with mode

    ('cat_encoder', OneHotEncoder())                      # convert text to numbers

])
# Full pipeline

from sklearn.compose import ColumnTransformer



num_attribs_all = X_train.select_dtypes(['float64', 'int64']).columns

num_attribs = num_attribs_all.drop('PassengerId')



cat_attribs_all = X_train.select_dtypes('object').columns

cat_attribs = cat_attribs_all.drop(['Ticket', 'Cabin', 'Name'])



full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),

        ("cat", cat_pipeline, cat_attribs),

    ])
from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier





classifiers = [

    SGDClassifier(),

    LogisticRegression(),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()

    ]



for classifier in classifiers:

    pipe = Pipeline(

        [('full_pipeline', full_pipeline), 

         ('classifier', classifier)

        ])

    pipe.fit(X_train, y_train)   

    print(classifier)

    print("Accuracy: %.3f" % pipe.score(X_train, y_train))
knn = Pipeline([

    ('full_pipeline', full_pipeline),

    ('classifier', KNeighborsClassifier(3))

])



decision_tree = Pipeline([

    ('full_pipeline', full_pipeline),

    ('classifier', DecisionTreeClassifier())

])



random_forest = Pipeline([

    ('full_pipeline', full_pipeline),

    ('classifier', RandomForestClassifier())

])



ada_boost = Pipeline([

    ('full_pipeline', full_pipeline),

    ('classifer', AdaBoostClassifier())

])



gradient_boosting = Pipeline([

    ('full_pipeline', full_pipeline),

    ('classifier', GradientBoostingClassifier())

])



knn.fit(X_train, y_train)

decision_tree.fit(X_train, y_train)

random_forest.fit(X_train, y_train)

ada_boost.fit(X_train, y_train)

gradient_boosting.fit(X_train, y_train);
from sklearn.model_selection import cross_val_score



classifiers = [knn, decision_tree, random_forest, ada_boost, gradient_boosting]



labels = [

    'KNeighborsClassifier(3)',

    'DecisionTreeClassifier()',

    'RandomForestClassifier()',

    'AdaBoostClassifier()',

    'GradientBoostingClassifier()'

    ]



print('Cross-validation:')

i = 0

for classifier in classifiers:

    accuracy_cv = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')

    print(labels[i])

    print(np.mean(accuracy_cv))

    i += 1
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import roc_auc_score



print('ROC AUC scores:')

i = 0

for classifier in classifiers:

    y_pred = cross_val_predict(classifier, X_train, y_train)

    print(labels[i])

    print(roc_auc_score(y_train, y_pred))

    i += 1
from sklearn.metrics import precision_score

from sklearn.model_selection import cross_val_predict



print('Precision:')

i = 0

for classifier in classifiers:

    y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=5)

    print(labels[i])

    print(precision_score(y_train, y_train_pred))

    i +=1
from sklearn.metrics import recall_score

from sklearn.model_selection import cross_val_predict



print('Recall:')

i = 0

for classifier in classifiers:

    y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=5)

    print(labels[i])

    print(recall_score(y_train, y_train_pred))

    i +=1
from sklearn.metrics import f1_score

from sklearn.model_selection import cross_val_predict



print('F₁ scores:')

i = 0

for classifier in classifiers:

    y_train_pred = cross_val_predict(classifier, X_train, y_train, cv=5)

    print(labels[i])

    print(f1_score(y_train, y_train_pred))

    i += 1