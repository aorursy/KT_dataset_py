import pandas as pd

import numpy as np

import random as rnd





import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline





from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import make_pipeline, Pipeline



import category_encoders as ce



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
train_df
target = 'Survived'

y = train_df[target]

train_df = train_df.drop([target], axis = 1)
train_df.info()
numerical_cols = [col for col in train_df.columns if train_df[col].dtype in ['int64', 'float64']]

categorical_cols = [col for col in train_df.columns if train_df[col].dtype == "object"]
categorical_cols
numerical_cols
train_df.isnull().any()
train_df['Cabin'] = train_df['Cabin'].fillna(value = 'N/A')

train_df['Age']= train_df['Age'].fillna(train_df['Age'].mean())

train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode().iloc[0])

train_df.isnull().any()
train_df['Name'].value_counts()
train_df['Sex'].value_counts()
train_df['Ticket'].value_counts()
train_df['Cabin'].value_counts()
train_df['Embarked'].value_counts()
#df_cat = train_df[categorical_cols]
X_train, X_valid, y_train, y_valid = train_test_split(train_df, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)
X_train
categorical_transformer = Pipeline(steps=[

    ('label', ce.ordinal.OrdinalEncoder())

])
preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])
clf = LogisticRegression(class_weight='balanced', random_state = 777)
pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])
cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')
pipe.fit(X_train, y_train)
y_preds = pipe.predict(X_valid)
acc = accuracy_score(y_valid, y_preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)

categorical_transformer = Pipeline(steps=[

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)
categorical_transformer = Pipeline(steps=[

    ('target', ce.target_encoder.TargetEncoder(handle_unknown='value'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)
categorical_transformer = Pipeline(steps=[

    ('woe', ce.woe.WOEEncoder(handle_unknown='value'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)
categorical_transformer = Pipeline(steps=[

    ('james-stein', ce.james_stein.JamesSteinEncoder(handle_unknown='value'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)
categorical_transformer = Pipeline(steps=[

    ('james-stein', ce.leave_one_out.LeaveOneOutEncoder(handle_unknown='value'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)
categorical_transformer = Pipeline(steps=[

    ('james-stein', ce.cat_boost.CatBoostEncoder(handle_unknown='value'))

])



preprocess = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



clf = LogisticRegression(class_weight='balanced', n_jobs=1, random_state = 777)



pipe = Pipeline(steps = [('preprocess', preprocess),

                               ('clf', clf )])



cv_score = cross_val_score(pipe, X_train, y_train, cv=5, n_jobs=-1, scoring='accuracy')



pipe.fit(X_train, y_train)



preds = pipe.predict(X_valid)



acc = accuracy_score(y_valid, preds)

print('CV Mean Accuracy :', cv_score.mean())

print('CV Standard Deviation:', cv_score.std())

print('Validation Accuracy:', acc)