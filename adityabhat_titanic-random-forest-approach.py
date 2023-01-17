import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.base import clone

import matplotlib.pyplot as plt

from sklearn.model_selection import * 

from sklearn.metrics import *

from sklearn.preprocessing import *

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import *
train = pd.read_csv('/kaggle/input/titanic/train.csv', low_memory=False)

test = pd.read_csv('/kaggle/input/titanic/test.csv', low_memory=False)
def classification_metrics(y_true,y_pred):

    """A function to print Evaluation metrics"""

    d = {

        'Accuracy': accuracy_score(y_true, y_pred),

        'Precision': precision_score(y_true, y_pred),

            'Recall': recall_score(y_true, y_pred),

            'F1': f1_score(y_true, y_pred),

            'AUC': roc_auc_score(y_true, y_pred),

            'Log Loss': log_loss(y_true, y_pred)

             }

    d = {k: np.round(v, 4) for k, v in d.items()}

    print(d)
def custom_cross_val(clf, X_train, y_train, features=None):

    skfolds = StratifiedKFold(n_splits=5)

    

    for n_fold, (train_idx, test_idx) in enumerate(skfolds.split(X_train, y_train)):

        print('Evaluating fold number: {}'.format(n_fold) )

        clone_clf = clone(clf)

        X_train_folds = X_train[features].iloc[train_idx]

        y_train_folds = y_train.iloc[train_idx]

        X_test_folds = X_train[features].iloc[test_idx]

        y_test_folds = y_train.iloc[test_idx]

        

        clone_clf.fit(X_train_folds, y_train_folds)

        y_pred = clone_clf.predict(X_test_folds)

        classification_metrics(y_test_folds, y_pred)

    
t = train.head(20)

p = t.pop('Survived')

x = StratifiedKFold(n_splits=4)

for n_fold, (a, b) in enumerate(x.split(t, p)):

    print(n_fold)

    #print(t.iloc[a])

    print(p.iloc[b])
def cross_validation(clf, X, y, scoring='accuracy', cv=10):

    """"A function to print Cross Validation metrics."""

    

    scores = cross_val_score(clf, X, y, scoring=scoring, cv=cv)

    return {

        'Scores': scores,

        'Mean': scores.mean(),

        'Standard Deviation': scores.std()

    }
"""

Write a function that take a list of columns and an imputation strategy to fit. The function returns an imputer object which will then be 

used to transform the train and test data.  

"""
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train['LastName'] = train['Name'].str.split(r",", expand=True, n=1).get(0)

test['LastName'] = test['Name'].str.split(r",", expand=True, n=1).get(0)

train['Title'] = train.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)

test['Title'] = test.Name.str.extract(r',\s*([^\.]*)\s*\.', expand=False)
def title_transform(x):

    if x == 'Mr':

        return x

    elif x in ['Mrs', 'Miss', 'Mme','Ms','Lady', 'Mlle', 'the Countess']:

        return 'Ms'

    elif x == 'Master':

        return x

    else:

        return 'Rare'
train['Title'] = train.Title.apply(title_transform)

test['Title'] = test.Title.apply(title_transform)
train.head()
train['has_cabin'] = train.Cabin.notna().astype(int)

test['has_cabin'] = test.Cabin.notna().astype(int)
train.head()
train['family_size'] = train['SibSp']+ train['Parch'] 

test['family_size'] = test['SibSp']+ test['Parch'] 
train['is_alone'] = train['family_size'].eq(0).astype(int)

test['is_alone'] = test['family_size'].eq(0).astype(int)
features_to_encode = ['Sex', 'Title', 'Embarked']
train[features_to_encode] = train[features_to_encode].fillna('missing')
test[features_to_encode] = test[features_to_encode].fillna('missing')
encoded_features = ['sex_enc', 'title_enc', 'embarked_enc']
train[encoded_features] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=train.index)

test[encoded_features] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=test.index)
ordinal_encoder = OrdinalEncoder()

train[encoded_features] = ordinal_encoder.fit_transform(train[features_to_encode])

test[encoded_features] = ordinal_encoder.transform(test[features_to_encode])
ordinal_encoder.categories_
y = train['Survived']

X = train.drop(columns=['PassengerId','Survived'])
# Split the data 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train.head()
# Fare Imputer

fare_imputer = SimpleImputer(strategy='median', copy=True, verbose=0)

X_train['imputed_fare'] = fare_imputer.fit_transform(X_train.loc[:, ['Fare']])

X_test['imputed_fare'] = fare_imputer.transform(X_test.loc[:, ['Fare']])

test['imputed_fare'] = fare_imputer.transform(test.loc[:, ['Fare']])
# Embarked Imputer



emb_imputer = SimpleImputer(strategy='most_frequent', missing_values=3.0, copy=True)

X_train['imp_embarked'] = emb_imputer.fit_transform(X_train.loc[:, ['embarked_enc']])

X_test['imp_embarked'] = emb_imputer.transform(X_test.loc[:, ['embarked_enc']])

test['imp_embarked'] = emb_imputer.transform(test.loc[:, ['embarked_enc']])
# Age median Imputer

age_imputer = SimpleImputer(strategy='median')

X_train['imp_med_age'] = age_imputer.fit_transform(X_train.loc[:, ['Age']])

X_test['imp_med_age'] = age_imputer.transform(X_test.loc[:, ['Age']])

test['imp_med_age'] = age_imputer.transform(test.loc[:, ['Age']])
test.head()
# Lets look at the features we have in the data. 

X_train.info()
features = ['Pclass','sex_enc','SibSp','Parch']
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train[features] ,y_train)
y_train_pred = clf.predict(X_train[features])

classification_metrics(y_train,y_train_pred)
y_test_pred = clf.predict(X_test[features])

classification_metrics(y_test,y_test_pred)
cross_val_score(clf, X_train[features], y_train, cv=5)
X_train.info()
features = ['Pclass','SibSp','Parch', 'has_cabin', 'family_size', 'is_alone','sex_enc', 'title_enc', 'imputed_fare', 'imp_embarked','imp_med_age']
clf = RandomForestClassifier(n_jobs=-1)

clf.fit(X_train[features] ,y_train)
y_train_pred = clf.predict(X_train[features])

classification_metrics(y_train,y_train_pred)
y_test_pred = clf.predict(X_test[features])

classification_metrics(y_test,y_test_pred)
custom_cross_val(clf, X_train, y_train, features=features)
cross_val_score(clf, X_train[features], y_train, cv=5)
clf_1 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=3, max_features='sqrt' ,n_jobs=-1)

clf_1.fit(X_train[features] ,y_train)
y_train_pred = clf_1.predict(X_train[features])

classification_metrics(y_train,y_train_pred)
y_test_pred = clf_1.predict(X_test[features])

classification_metrics(y_test,y_test_pred)
cross_val_score(clf_1, X_train[features], y_train, cv=5)
custom_cross_val(clf_1, X_train, y_train, features=features)
# Feature Importances

feature_scores = pd.Series(clf_1.feature_importances_, index=X_train[features].columns.to_list()).sort_values(ascending=False)



# Creating a seaborn bar plot



f, ax = plt.subplots(figsize=(20, 10))

ax = sns.barplot(x=feature_scores, y=feature_scores.index)

ax.set_title("Visualize feature scores of the features")

ax.set_yticklabels(feature_scores.index)

ax.set_xlabel("Feature importance score")

ax.set_ylabel("Features")

plt.show()
y_eval = clf_1.predict(test[features])
test['Survived'] = y_eval

df_sub = test.loc[:,['PassengerId', 'Survived']]

df_sub.to_csv('2.csv', mode = 'w', index=False)