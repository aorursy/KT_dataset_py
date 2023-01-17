import numpy as np

import pandas as pd

import math

import seaborn as sns

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

from sklearn.inspection import plot_partial_dependence

%matplotlib inline

sns.set_style('whitegrid')



# Processing

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, RobustScaler

import category_encoders as ce # Count, Target, CatBoost



# Models

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,

        ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier)

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import (Perceptron, SGDClassifier,

        LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier)

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# Evaluation

from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, learning_curve

from sklearn.metrics import accuracy_score



raw = pd.read_csv('/kaggle/input/iris/Iris.csv')



# drop PetalLengthCm because of high correlation with PetalWidthCm.

# I chose to drop PetalLengthCm instead of PetalWidthCm because PetalWidthCm had a bit less

# correlation with other features. I used correlation heatmaps and pairplots to explore this.

raw.drop(['Id', 'PetalLengthCm'], axis=1, inplace=True)

le = LabelEncoder()

raw['Species'] = le.fit_transform(raw['Species'])



X_trn, X_tst, y_trn, y_tst = train_test_split(raw.drop('Species', axis=1), raw['Species'], test_size=0.25, random_state=420)

X_all = [X_trn, X_tst]



def correlation_heatmap(df):

    plt.figure(figsize=(20, 10))

    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(df.corr(), cmap=colormap, annot=True)

    plt.title('Pearson Correlation of Features', size=15)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)



estimators = [

    KNeighborsClassifier(),

    NearestCentroid(),

    RadiusNeighborsClassifier(),

    SVC(),

    LinearSVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    BaggingClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    BernoulliNB(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    Perceptron(),

    SGDClassifier(),

    LogisticRegression(),

    PassiveAggressiveClassifier(),

    RidgeClassifier(),

    GaussianProcessClassifier(),

    MLPClassifier(),

    XGBClassifier(),

    LGBMClassifier(),

]
scores = pd.DataFrame(columns=['Estimator', 'Test Score', 'Test Score 3*STD'])

preds = pd.DataFrame(y_trn)

for i in range(len(estimators)):

    est = estimators[i]

    est_name = est.__class__.__name__

#     RFE

#     if hasattr(est, 'coef_') or hasattr(est, 'feature_importances_'):

#         est = RFECV(est, scoring='accuracy', cv=skf)

#         est.fit(train[feature_cols], train[targ_col])

#         rfe_cols = feature_cols.values[est.support_]

    est.fit(X_trn, y_trn)

    cv_scores = cross_val_score(est, X_trn, y_trn, cv=skf, n_jobs=-1)

#     cv_scores = cross_val_score(model, train[rfe_cols[clf_name] if hasattr(clf, 'coef_') or hasattr(clf, 'feature_importances_') else feature_cols], train[targ_col], cv=skf)

    scores.loc[i] = [est_name, cv_scores.mean(), cv_scores.std() * 3]

    preds[est_name] = est.predict(X_trn)

scores.sort_values(by='Test Score', ascending=False, inplace=True)

plt.figure(figsize=(20, 10))

sns.barplot(x=scores['Test Score'], y=scores['Estimator'])
correlation_heatmap(preds)
vote_eligible = [

    est.__class__.__name__ for est in estimators

    if hasattr(est, 'predict_proba')

    and not est.__class__.__name__ in [

        'BernoulliNB', 'DecisionTreeClassifier', 'RandomForestClassifier', 'BaggingClassifier',

        'ExtraTreesClassifier', 'GradientBoostingClassifier', 'MLPClassifier', 'LogisticRegression',

        'AdaBoostClassifier', 'GaussianProcessClassifier', 'LGBMClassifier'

    ]

]

print(vote_eligible)

correlation_heatmap(preds[vote_eligible])
def print_scores_info(model_name, scores):

    mean = scores.mean() * 100

    std_3 = scores.std() * 100 * 3

    print(model_name, 'score mean: ', mean)

    print(model_name, 'score 3 std range: ', mean - std_3, '—', mean + std_3)



vote_estimators = [

    ('knn', KNeighborsClassifier()),

    ('rnc', RadiusNeighborsClassifier()),

    ('gnb', GaussianNB()),

    ('lda', LinearDiscriminantAnalysis()),

    ('qda', QuadraticDiscriminantAnalysis()),

    ('xgb', XGBClassifier(**{'booster': 'gbtree', 'learning_rate': 0.1, 'max_depth': 10, 'n_jobs': -1, 'random_state': 0, 'reg_alpha': 0.16, 'reg_lambda': 2.56})),

]



vote_hard_est = VotingClassifier(vote_estimators, voting='hard')



print_scores_info('Vote Hard', cross_val_score(vote_hard_est, X_trn, y_trn, cv=skf, n_jobs=-1))



vote_hard_est.fit(X_trn, y_trn)

preds_test = vote_hard_est.predict(X_tst)

accuracy_score(y_tst, preds_test)
# 0.9736842105263158 test accuracy, train_test_split seed as 0

# 1.0 test accuracy, train_test_split seed as 420
def plot_learning_curve(model, X, y, title, cv, train_sizes):

    train_sizes, train_scores, test_scores = learning_curve(

        model, X, y, cv=cv, n_jobs=-1, random_state=0, train_sizes=train_sizes

    )

    

    plt.figure(figsize=(14, 8))

    plt.title(title)

    ylim = (0.7, 1.01)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                train_scores_mean + train_scores_std, alpha=0.1,

                color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

        label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

        label="Cross-validation score")



    plt.legend(loc="best")

    plt.show()



plot_learning_curve(vote_hard_est, X_trn, y_trn,

                    'Vote Hard Model Learning Curve', skf, np.linspace(0.2, 1.0, 5))