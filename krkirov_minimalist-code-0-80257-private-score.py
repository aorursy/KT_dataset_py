import numpy as np

import pandas as pd

import copy



from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from sklearn.linear_model import LogisticRegressionCV
def ml_pipe(classifier):  

    pl = Pipeline([

            ('ohe', OneHotEncoder(categories='auto',

                                  handle_unknown='ignore')),

            ('classifier', classifier)

              ])

    return pl



def deploy_pipe(pl, X_tr, y_tr, X_valid, y_valid):

    """ """

    pl_hist = pl.fit(X_tr, y_tr)

    y_pred = pl.predict(X_valid)

    pl_acc = pl.score(X_valid, y_valid)

    print('\nModel Accuracy Score: {} \n'.format(pl_acc))

    return pl, pl_hist, pl_acc
# Load train and test data

df_train = pd.read_csv('../input/cat-in-the-dat/train.csv', index_col=0)

y = df_train['target'].values.ravel()

df_train.drop(['target'], axis=1, inplace=True)

df_train = df_train.applymap(str)



df_test = pd.read_csv('../input/cat-in-the-dat/test.csv', index_col=0)

df_test = df_test.applymap(str)
# Feature engineering

nom_cols = [col for col in df_train.columns if 'nom' in col]

df_train['nom'] = df_train[nom_cols].apply('_'.join, axis=1)

df_test['nom'] = df_test[nom_cols].apply('_'.join, axis=1)



bin_cols = [col for col in df_train.columns if 'bin' in col]

df_train['bin'] = df_train[nom_cols].apply('_'.join, axis=1)

df_test['bin'] = df_test[nom_cols].apply('_'.join, axis=1)
# Specify and train the model

classifier = LogisticRegressionCV(Cs=np.linspace(0.11, 0.14, num=100), max_iter=500, cv=10, n_jobs=-1)

pl = ml_pipe(classifier)

pl, pl_history, pl_accuracy = deploy_pipe(pl, df_train, y, df_train, y)
df_submit = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv', index_col=0)

df_submit['target'] = pl.predict_proba(df_test)[:, 1]

df_submit.to_csv('cat_logreg_cv.csv')
# Best regularisation constant

pl['classifier'].C_