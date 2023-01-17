# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.offline as py



from collections import Counter

from sklearn import metrics

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn import linear_model

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from lightgbm import LGBMClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier



# Read the train data set

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.head())



# Check for null value

train.isnull().any().any()

print(train.isnull().any().any())



Counter(train.dtypes.values)

print(Counter(train.dtypes.values))



# Check for missing vales by its features

train2= (train.isnull().sum() / len(train)) * 100

misval = train2.drop(train2[train2 == 0].index).sort_values(ascending=False)[:30]

missing = pd.DataFrame({'Missing %' :misval})

missing.head(10)

print(missing.head(10))



# Phishing out potentential values probably having -1

train_copy = train.replace(-1, np.NaN)

train_copy= (train_copy.isnull().sum() / len(train_copy)) * 100

train_copy = train_copy.drop(train_copy[train_copy == 0].index).sort_values(ascending=False)[:30]

missing = pd.DataFrame({'Missing %' :train_copy})

missing.head(10)

print(missing.head(10))



# group to either intger or floating data type

train_float = train.select_dtypes(include=['float64'])

train_int = train.select_dtypes(include=['int64'])

print(train_float)

print(train_int)



# group by feature types

bin_col = [col for col in train.columns if '_bin' in col] #binary

cat_col = [col for col in train.columns if '_cat' in col] #categorical

# group by numerical features

num_col = [x for x in train.columns if x[-3:] not in ['bin', 'cat']]

# group by individual, car, region and calculated fields

ind_col = [col for col in train.columns if '_ind_' in col] #individual

car_col = [col for col in train.columns if '_car_' in col] #car

reg_col = [col for col in train.columns if '_reg_' in col] #region

calc_col = [col for col in train.columns if '_calc_' in col] #calculation



zero_list = []

one_list = []

for col in bin_col:

    zero_list.append((train[col] == 0).sum())

    one_list.append((train[col] == 1).sum())

print(zero_list)

print(one_list)



# Corralation matrix

cor_matrix = train[num_col].corr().round(2)

print(cor_matrix)



# Correlation of float values

colormap = plt.cm.magma

plt.figure(figsize=(16,12))

plt.title('Correlation of float features', y=1.05, size=15)

sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True,

            cmap=colormap, linecolor='white', annot=True)

plt.show()



tot_cat_col = list(train.select_dtypes(include=['category']).columns)



other_cat_col = [c for c in tot_cat_col if c not in cat_col+ bin_col]

other_cat_col



# Using PCA

X = train.drop(['id', 'target'], axis=1).values

y = train['target'].values.astype(np.int8)



# Standardize feature

X_scaled = preprocessing.scale(X)

print(X)



target_names = np.unique(y)

print('\nThere are %d unique target valuess in this dataset:' % (len(target_names)), target_names)

n_comp = 10



# PCA

print('\nRunning PCA ...')

pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)

X_pca = pca.fit_transform(X)

print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())



print('Individual variance contributions:')

for j in range(n_comp):

    print(pca.explained_variance_ratio_[j])



# Plotting the scatter plot of the training data on the 1st and 2nd PC

colors = ['orange', 'blue']

plt.figure(1, figsize=(11, 11))



for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=1,

                alpha=.7, label=target_name, marker='.')

plt.legend(loc='best', shadow=False, scatterpoints=3)

plt.title(

        "Training data projected on the 1st "

        "and 2nd principal components")

plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[0] * 100.0))

plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[1] * 100.0))

plt.show()



# Validation of the train data

num_folds = 8

seed = 8

scoring = 'Accuracy'



X = X_pca

Y = np.array(train['target'])



validation_size = 0.25

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=

                                                                validation_size, random_state=seed)

# generate results with linear algorithms

models = [('LR', LogisticRegression()),

          ('NB', GaussianNB())]

results =[]

names = []

for name, model in models:

    print("Training model %s" % (name))

    model.fit(X_train, Y_train)

    result = model.score(X_validation, Y_validation)

    info = "Classifier score %s: %f" %(name, result)

    print(info)

print("Done")



# Trying a boosting method using the python package Light Gradient Boosting Method(LGBM)

id_test = test['id'].values

target_train = train['target'].values



train = train.drop(['target','id'], axis = 1)

test = test.drop(['id'], axis = 1)



col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]

train = train.drop(col_to_drop, axis=1)

test = test.drop(col_to_drop, axis=1)



train = train.replace(-1, np.nan)

test = test.replace(-1, np.nan)



cat_features = [a for a in train.columns if a.endswith('cat')]



for column in cat_features:

    temp = pd.get_dummies(pd.Series(train[column]))

    train = pd.concat([train, temp], axis=1)

    train = train.drop([column], axis=1)



for column in cat_features:

    temp = pd.get_dummies(pd.Series(test[column]))

    test = pd.concat([test, temp], axis=1)

    test = test.drop([column], axis=1)



print(train.values.shape, test.values.shape)



class Ensemble(object):

    def __init__(self, n_splits, stacker, base_models):

        self.n_splits = n_splits

        self.stacker = stacker

        self.base_models = base_models



    def fit_predict(self, X, y, T):

        X = np.array(X)

        y = np.array(y)

        T = np.array(T)



        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=2016).split(X, y))



        S_train = np.zeros((X.shape[0], len(self.base_models)))

        S_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):



            S_test_i = np.zeros((T.shape[0], self.n_splits))



            for j, (train_idx, test_idx) in enumerate(folds):

                X_train = X[train_idx]

                y_train = y[train_idx]

                X_holdout = X[test_idx]



                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))

                clf.fit(X_train, y_train)

                y_pred = clf.predict_proba(X_holdout)[:,1]



                S_train[test_idx, i] = y_pred

                S_test_i[:, j] = clf.predict_proba(T)[:,1]

            S_test[:, i] = S_test_i.mean(axis=1)



        results = cross_val_score(self.stacker, S_train, y, cv=3, scoring='roc_auc')

        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)

        res = self.stacker.predict_proba(S_test)[:, 1]

        return res



# LightGBM params

lgb_params = {}

lgb_params['learning_rate'] = 0.02

lgb_params['n_estimators'] = 650

lgb_params['max_bin'] = 10

lgb_params['subsample'] = 0.8

lgb_params['subsample_freq'] = 10

lgb_params['colsample_bytree'] = 0.8

lgb_params['min_child_samples'] = 500

lgb_params['seed'] = 99



lgb_params2 = {}

lgb_params2['n_estimators'] = 1090

lgb_params2['learning_rate'] = 0.02

lgb_params2['colsample_bytree'] = 0.3

lgb_params2['subsample'] = 0.7

lgb_params2['subsample_freq'] = 2

lgb_params2['num_leaves'] = 16

lgb_params2['seed'] = 99



lgb_params3 = {}

lgb_params3['n_estimators'] = 1100

lgb_params3['max_depth'] = 4

lgb_params3['learning_rate'] = 0.02

lgb_params3['seed'] = 99



lgb_model = LGBMClassifier(**lgb_params)

lgb_model2 = LGBMClassifier(**lgb_params2)

lgb_model3 = LGBMClassifier(**lgb_params3)



log_model = LogisticRegression()



stack = Ensemble(n_splits=3,

                 stacker=log_model,

                 base_models=(lgb_model, lgb_model2, lgb_model3))



y_pred = stack.fit_predict(train, target_train, test)



sub = pd.DataFrame()

sub['id'] = id_test

sub['target'] = y_pred

sub.to_csv('porto_output.csv', index=False)



# Any results you write to the current directory are saved as output.