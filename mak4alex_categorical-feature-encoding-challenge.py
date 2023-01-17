import os

import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.compose import ColumnTransformer 

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, LabelBinarizer, MinMaxScaler

from sklearn.model_selection import cross_validate, learning_curve, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier, RandomForestClassifier

from sklearn.utils import shuffle

import matplotlib.pyplot as plt



random_state = 42

np.random.seed(random_state)

cv = 4



for dirname, _, filenames in os.walk('input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%%time

train_df = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv', index_col='id')

sample_submission_df = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

test_df = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv', index_col='id')
sample_submission_df.head()
print(train_df.shape)

print(test_df.shape)
train_df.info()

train_df = shuffle(train_df, random_state=random_state)

train_Y = train_df['target'].copy()

train_X = train_df.drop('target', axis=1).copy()
train_df.head()
def get_pipline(estimator):

    return Pipeline(steps=[

        ('preproc', ColumnTransformer([

            ('bin_0_2', 'passthrough', ['bin_0', 'bin_1', 'bin_2']),

            ('bin_3_4', FunctionTransformer(func=lambda X: X.replace({'F': 0, 'T': 1, 'N': 0, 'Y': 1}), validate=False), [

                'bin_3', 'bin_4']

            ),

            ('nom_0_4', OneHotEncoder(sparse=True, handle_unknown='ignore'), [

                'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

            ),

            ('ord', Pipeline(steps=[

                ('replace', ColumnTransformer([

                    ('encoder',  OrdinalEncoder(categories=[

                        ['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'],

                        ['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'],

                        np.sort(train_df['ord_3'].unique()),

                        np.sort(train_df['ord_4'].unique()),

                        np.sort(train_df['ord_5'].unique()),

                    ]), ['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']),

                ], remainder='passthrough')),

                ('mm_scaler', MinMaxScaler())

            ]), ['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'] )

        ])),    

        ('est', estimator)

    ])



xgb_pipline = get_pipline(XGBClassifier(objective='binary:logistic', n_estimators=1100, max_depth=6, gamma=5))

logit_pipline = get_pipline(LogisticRegression(solver='lbfgs', max_iter=225, C=0.12, random_state=random_state))



pipline = VotingClassifier([('xgb', xgb_pipline), ('logit', logit_pipline)], voting='soft', n_jobs=-1)
params = {'est__solver': ['lbfgs'], 'est__C': [0.11, 0.12, 0.13], 'est__max_iter': [225, 250, 275]}

gs_cv = GridSearchCV(logit_pipline, params, scoring='roc_auc', cv=cv, n_jobs=-1, return_train_score=True, verbose=1)



#print('Best params {}, score {}'.format(gs_result.best_params_, gs_result.best_score_))

#Best params {'est__C': 0.12, 'est__max_iter': 225, 'est__solver': 'lbfgs'}, score 0.7998486767216677
cv_scores = cross_validate(

    pipline, train_X, train_Y, scoring='roc_auc', cv=cv, n_jobs=-1, 

    return_train_score=True, return_estimator=True, verbose=1

)
print(cv_scores['train_score'])

print(cv_scores['test_score'])

cv_scores.keys()
%%time



scores = np.array([est.predict_proba(test_df) for est in cv_scores['estimator']])

mean_scores = scores.mean(axis=0)[:, 0]



submit_df = pd.DataFrame({ 'id': test_df.index, 'target': mean_scores })

submit_df.to_csv('submission.csv', index=False)
def plot_learning_curve(estimator, title, X, y):

    plt.figure()

    plt.title(title)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), 

        random_state=random_state)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1, color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")



    plt.legend(loc="best")

    plt.show()



plot_learning_curve(logit_pipline, 'logit', train_X, train_Y)