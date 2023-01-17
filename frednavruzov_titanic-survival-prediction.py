# disable IPython warnings

import warnings

warnings.filterwarnings('ignore')



# import required libraries/packages

import pandas as pd

import numpy as np



# models

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB



# utility / measurements

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_extraction import FeatureHasher

from sklearn.preprocessing import Binarizer, OneHotEncoder, StandardScaler, FunctionTransformer

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV

from sklearn.base import TransformerMixin

from sklearn.metrics import accuracy_score



# allow inline plotting

import matplotlib.pyplot as plt

%matplotlib inline
# import data and take a first look on it

df_train = pd.read_csv('../input/train.csv', sep=',', header=0)

df_test = pd.read_csv('../input/test.csv', sep=',', header=0)

print('train DF shape: {}'.format(df_train.shape))

df_train.head()
print(df_train.count())

print('\nTarget class balance:\ndied: {}\nsurvived: {}'.format(*df_train['Survived'].value_counts().values))

df_train.describe()
def clean_data(df, outlier_columns=None):

    """

    Function that cleans input data and transforms some existing features into new ones

    

    - **parameters**, **types**, **return** and **return types**::

        :param df: dataframe that holds initial unprocessed data

 

        :return: return transformed and augmented copy of initial DataFrame

        :rtype: pd.DataFrame object

    """

    

    if type(df) != type(pd.DataFrame()):

        raise TypeError('please use a pandas.DataFrame object as input')

    

    pdf = df.copy() # make a copy to make all changes do not touch initial DataFrame

    

    # 1.fillna and missing data

    

    # numerical features

    nulls = pdf.select_dtypes(include=[np.number]).isnull().astype(int).sum()

    for i in nulls.index:

        if nulls[i] > 0:

            # group data by gender and pclass

            ms = df.dropna().groupby(['Sex', 'Pclass']).median()[i] # group medians



            d = {(i1, i2): ms.loc[(ms.index.get_level_values('Sex') == i1) &

                                  (ms.index.get_level_values('Pclass') == i2)].values[0]

                 for i1 in ms.index.levels[0] for i2 in ms.index.levels[1]}



            pdf['median'] = pdf.apply(lambda row: d[(row['Sex'], row['Pclass'])], axis=1)

            pdf[i].fillna(pdf['median'], inplace=True)



    # categorical features

    nulls = df.select_dtypes(exclude=[np.number]).isnull().astype(int).sum()

    for i in nulls.index:

        if nulls[i] > 0 and i == 'Cabin':

            pdf[i].fillna('1', inplace=True)

        elif nulls[i] > 0:

            # group data by gender and pclass

            ms = pdf.dropna().groupby(['Sex', 'Pclass'])[i].agg(lambda x:x.value_counts().index[0]) # group modes



            d = {(i1, i2): ms.loc[(ms.index.get_level_values('Sex') == i1) & 

                                  (ms.index.get_level_values('Pclass') == i2)].values[0]

                 for i1 in ms.index.levels[0] for i2 in ms.index.levels[1]}



            pdf['mode'] = pdf.apply(lambda row: d[(row['Sex'], row['Pclass'])], axis=1)

            pdf[i].fillna(pdf['mode'], inplace=True)

    

    # 3. extract additional features

    # DECK ----------------------------------------

    pdf['Deck'] = pdf['Cabin'].str.lower().str[0]

    # Title ---------------------------------------

    pdf['Title'] = pdf['Name'].str.replace('(.*, )|(\\..*)', '').str.lower()

    rare_titles = ['dona', 'lady', 'the countess','capt', 'col', 'don', 'dr', 'major', 'rev', 'sir', 'jonkheer']

    ud = dict.fromkeys(rare_titles, 'rare title'); ud.update({'mlle':'miss', 'ms':'miss', 'mme':'mrs'})# merge titles

    pdf['Title'] = pdf['Title'].replace(ud)

    # IsChild -------------------------------------

    pdf['IsChild'] = ((pdf['Age'] < 18) & (pdf['Title'].isin(['master', 'miss']))).astype(int)

    # IsMother -------------------------------------

    pdf['IsMother'] = ((pdf['Age'] > 18) & (pdf['Title'] == 'mrs') & (pdf['Parch'] > 0)).astype(int)

    

    # 3. transform old features

    pdf['IsMale'] = pdf['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    pdf['Embarked'] = pdf['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)

    

    

    pdf['Title'] = pdf['Title'].map({'miss':1,

                                     'mrs':2,

                                     'master':3,

                                     'mr':4,

                                     'rare title':5

                                    }).astype(int)# to ints

    

    pdf['Deck'] = pdf['Deck'].map(dict(zip('1abcdefgt', range(0,9)))).astype(int) # map to ints

    

    

    # 4. substitute outliers (for numerical columns)

    if outlier_columns:

        for c in outlier_columns:

            q = pdf[c].quantile([0.2, 0.8]).values

            pdf[c] = pdf[c].apply(lambda x: q[0] if x < q[0] else min(q[1], x))

    

    # 5. drop-redundant: drop useless features

    pdf.drop([

            'Cabin', 

            'Name', 

            'Sex', 

            'Ticket', 

            'median', 

            'mode',

            ], axis=1, inplace=True, errors='ignore')

     

    return pdf
train = clean_data(df_train, outlier_columns=['Fare'])

test = clean_data(df_test, outlier_columns=['Fare'])

print(train.count())

train.describe()
train.select_dtypes(include=[np.number]).drop(['PassengerId', 'IsMale', 'IsChild', 'IsMother'], errors='ignore', 

                                              axis=1).corr(method='pearson')
samples_to_add = train['Survived'].value_counts().values[0] - train['Survived'].value_counts().values[1]

add_survived = train[train['Survived'] == 1].sample(n=samples_to_add, replace=True)



train = pd.concat([train, add_survived], axis=0)

train['Survived'].value_counts()
# split train on train/holdout parts

print("initial train shape: {}".format(train.shape))



train_tr, train_ho, y_train, y_ho = train_test_split(train.drop(['Survived', 'PassengerId'], axis=1), 

                                                     train['Survived'].values,

                                                     test_size=0.3, 

                                                     stratify=train['Survived'].values, 

                                                     random_state=42)

print ("X train shape: {}, X holdout shape: {}".format(train_tr.shape, train_ho.shape))

print ("y train shape: {}, y holdout shape: {}".format(y_train.shape, y_ho.shape))



# define column types for proper transformation/encoding

binary_cols = ['IsMale', 'IsChild', 'IsMother']

categorical_cols = ['Pclass', 'Embarked', 'Deck', 'Title']

numeric_cols = set(train.columns) - set(['Survived', 'PassengerId'] + binary_cols + categorical_cols)



# making correspondent boolean column indices

bdata_indices = np.array([(col in binary_cols) for col in train_tr.columns], dtype=bool)

cdata_indices = np.array([(col in categorical_cols) for col in train_tr.columns], dtype=bool)

ndata_indices = np.array([(col in numeric_cols) for col in train_tr.columns], dtype=bool)
# simple class to get rid of sparse format, incompatible with some classifiers

class DenseTransformer(TransformerMixin):



    def transform(self, X, y=None, **fit_params):

        return X.todense()



    def fit_transform(self, X, y=None, **fit_params):

        self.fit(X, y, **fit_params)

        return self.transform(X)



    def fit(self, X, y=None, **fit_params):

        return self

    

    def get_params(self, deep=True):

        return dict()



# create pipeline of transformation/extraction steps + classification step

def make_pipe(classifier):

    pipe = Pipeline(

        steps = [

            ('feature_processing', FeatureUnion(

                    transformer_list = [

                        # binary data

                        ('binary_processing', FunctionTransformer(lambda x: x[:, bdata_indices])),



                        # categorical data

                        ('categorical_processing', Pipeline(steps = [

                                    ('selecting', FunctionTransformer(lambda x: x[:, cdata_indices])),

                                    #('label_encoding', LabelEncoder()),

                                    #('hot_encoding', FeatureHasher())

                                    ('hot_encoding', OneHotEncoder(handle_unknown='ignore'))

                                ]

                             )

                        ),



                        # numeric data

                        ('numeric_processing', Pipeline(steps = [

                                    ('selecting', FunctionTransformer(lambda x: x[:, ndata_indices])),

                                    ('scaling', StandardScaler(with_mean=0.))

                                ]

                             )

                        ),

                    ]

                )

            ),

            ('dense', DenseTransformer()),

            ('clf', classifier)

        ]

    )

    return pipe



# base classificators

clfs = [

    ('SGDClassifier', SGDClassifier(random_state=42)),

    ('LogisticRegression', LogisticRegression(random_state=42)),

    ('LinearSVC', LinearSVC(random_state=42)),

    ('KNN', KNeighborsClassifier(n_neighbors=10)),

    ('RandomForestClassifier', RandomForestClassifier(random_state=42)),

    ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),

    ('GaussianNB', GaussianNB()),

    ('MultinomialNB', MultinomialNB()),

]



# cross-validation

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # use it to preserve equal class balance in folds

scores = [] # to hold cross-validation scores for base estimators



for c in clfs:

    pipe = make_pipe(c[1])

    score  = cross_val_score(pipe, X=train_tr.values, y=y_train, cv=cv).mean()

    scores.append([c[0], score])



for s in sorted(scores, key=lambda x: x[1], reverse=True):

    pass

    print("model: {}, accuracy={}".format(*s))
%%time



# 1. Random Forest --------------------------------------------------------------------------------------------

estimator_rf = make_pipe(RandomForestClassifier(random_state=7, n_jobs=-1, oob_score=True))



# set params grid for GridSearch to search through

params_grid_rf = {

    # 'clf__bootstrap': [True, False],

    #'clf__criterion': ["gini", "entropy"],

    'clf__max_depth': [None, 2, 3, 4, 5],

    #'clf__max_features': ["sqrt", "log2", None],

    'clf__min_samples_leaf': [1, 3],

    'clf__n_estimators': [10, 20, 30, 50, 100],

    #'clf__warm_start': [True, False]

}



# perform randomized search (100 iterations) because no of combinations is rather huge (2*2*5*3*2*5*2 = 1200)

grid_cv_rf = GridSearchCV(estimator_rf, param_grid=params_grid_rf, scoring='accuracy', cv=5)

grid_cv_rf.fit(train_ho, y_ho) # fit it on hold-out sample



# 2. Gradient Boosting -----------------------------------------------------------------------------------------

estimator_gb = make_pipe(GradientBoostingClassifier(random_state=42))



# set params grid for GridSearch to search through

params_grid_gb = {

    #'clf__loss': ['exponential', 'deviance'],

    'clf__learning_rate': [0.01, 0.1, 0.5, 1, 10],

    #'clf__n_estimators': [50, 100, 200],

    'clf__max_depth': [2, 3],

    'clf__subsample': [0.25, 0.5, 1.0]

}



# perform usual GridSearch

grid_cv_gb = GridSearchCV(estimator_gb, param_grid=params_grid_gb, scoring='accuracy', cv=5)

grid_cv_gb.fit(train_ho, y_ho) # fit it on hold-out sample



# 3. SVC -------------------------------------------------------------------------------------------------------

estimator_svc = make_pipe(LinearSVC(random_state=42))



# set params grid for GridSearch to search through

params_grid_svc = {

    'clf__fit_intercept': [True, False],

    'clf__dual': [True, False],

    #'clf__loss': ["hinge", "squared_hinge"],

    #'clf__penalty': ['l1', 'l2'], # L-1, L-2 euclidean,

    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]

}



# perform usual GridSearch

grid_cv_svc = GridSearchCV(estimator_svc, param_grid=params_grid_svc, scoring='accuracy', cv=5)

grid_cv_svc.fit(train_ho, y_ho) # fit it on hold-out sample
# best params obtained

print(grid_cv_svc.best_params_)

print(grid_cv_gb.best_params_)

print(grid_cv_rf.best_params_)



# best score obtained

print(grid_cv_svc.best_score_)

print(grid_cv_gb.best_score_)

print(grid_cv_rf.best_score_)
# construct ensemble

ensemble = VotingClassifier(estimators=[

                                        ('rf', grid_cv_rf.best_estimator_), 

                                        ('gb', grid_cv_gb.best_estimator_), 

                                        ('svc', grid_cv_svc.best_estimator_),

                                       ], voting='hard')

# fit it to the whole train dataset

ensemble.fit(train.drop(['Survived', 'PassengerId'], axis=1), train['Survived'].values)



# make prediction on test

prediction = ensemble.predict(test.drop(['PassengerId'], axis=1)) # predict labels based on X_test



answers = pd.DataFrame({'Survived': prediction}, test['PassengerId']) # predict labels

answers.to_csv('titanic-submission-Navruzov.csv') # save to submission file