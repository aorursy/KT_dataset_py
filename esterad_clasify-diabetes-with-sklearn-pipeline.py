import numpy as np

import pandas as pd

from scipy import stats



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB



from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, train_test_split



from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()
profile = ProfileReport(df)

profile.to_file(output_file="diabetes_report.html")


df['Pregnancies_cat'] = np.where(df['Pregnancies'] < 2, 'A',

                                 np.where(df['Pregnancies'] > 4, 'C',

                                          'B'))

df['Pregnancies_cat'] = df['Pregnancies_cat'].astype('category')

df['Pregnancies_cat'].value_counts()
df['Glucose'] = np.where(df['Glucose'] < 1,

                         df['Glucose'].mean(),

                         df['Glucose'])

df['Glucose'].hist()
df['BloodPressure'] = np.where(df['BloodPressure'] == 0,

                               df['BloodPressure'].mean(),

                               df['BloodPressure'])

df['BloodPressure'].hist()
df['SkinThickness'] = np.where(df['SkinThickness'] == 0,

                               df['SkinThickness'].median(),

                               df['SkinThickness'])

df['log_SkinThickness'] = np.log(df['SkinThickness'])

df['log_SkinThickness'].hist()
df['BMI'] = np.where(df['BMI'] == 0,

                     df['BMI'].median(),

                     df['BMI'])

df = df[(df['BMI'] < 100)]

df['log_BMI'] = np.log(df['BMI'])

df['log_BMI'].hist()
df = df[(df['Age'] < 100)]

df['boxcox_Age'] = stats.boxcox(df['Age'])[0]

df['boxcox_Age'].hist()
df.columns
df[df['Insulin'] > -1 ]['Insulin'].hist()
rplcmts = [np.random.choice(df[df['Insulin'] > 0 ]['Insulin'].values, replace=True) for i in range(df.shape[0])]
df['rplcmts'] = rplcmts

df['Insulin_new'] = np.where(df['Insulin'] == 0,

                             df['rplcmts'],

                             df['Insulin'])

df['log_Insulin_new'] = np.log(df['Insulin_new'])

df['log_Insulin_new'].hist()
df.columns
X = df[['Glucose', 'BloodPressure', 'log_SkinThickness', 'log_Insulin_new',

        'log_BMI', 'boxcox_Age', 'Pregnancies_cat']]

y = df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.base import BaseEstimator, TransformerMixin



class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):

        self.dtype = dtype

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.select_dtypes(include=[self.dtype])

    

class StringIndexer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.apply(lambda s: s.cat.codes.replace(

            {-1: len(s.cat.categories)}

        ))
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, StandardScaler



transformer = Pipeline([

    ('features', FeatureUnion(n_jobs=1, transformer_list=[

        

        # Part 1

        ('boolean', Pipeline([

            ('selector', TypeSelector('bool')),

        ])),  # booleans close

        

        ('numericals', Pipeline([

            ('selector', TypeSelector(np.number)),

            ('scaler', StandardScaler()),

        ])),  # numericals close

        

        # Part 2

        ('categoricals', Pipeline([

            ('selector', TypeSelector('category')),

            ('labeler', StringIndexer()),

            ('encoder', OneHotEncoder(handle_unknown='ignore')),

        ]))  # categoricals close

    ])),  # features close

])  # pipeline close
import warnings



warnings.simplefilter('ignore', FutureWarning)
models = [

    [('transformer', transformer), ('LogReg', LogisticRegression())],

#     [('transformer', transformer), ('RandomForest', RandomForestClassifier())],

    [('transformer', transformer),

#      ('RFE', RFE(LogisticRegression())),

     ('Bagging', BaggingClassifier(DecisionTreeClassifier(), max_features=5,

                                   max_samples=0.7,

                                   bootstrap=True))],

    [('transformer', transformer),

#      ('RFE', RFE(LogisticRegression())),

     ('SVC', SVC())]

]



params_grid = [

    {'LogReg__C': [0.001, 0.01, 0.1, 1, 10, 100], 'LogReg__penalty': ['l1', 'l2']},

#     {},

    {'Bagging__n_estimators': [50, 100], 

     'Bagging__base_estimator__max_depth': [5, 10, 20],

     'Bagging__base_estimator__min_samples_leaf': [15, 20]},

    [

        {'SVC__kernel': ['rbf'],

         'SVC__gamma': [1, 1e-1, 1e-2, 5e-2, 15e-2, 1e-3], 'SVC__C': [0.1, 1, 10, 50, 100, 150, 1000]},

        {'SVC__kernel': ['poly'],

         'SVC__degree': [3], 'SVC__coef0': [-0.1, 0, 0.05, 0.1, 0.15, 0.2, 1], 'SVC__C': [100, 150, 200, 1e3]}

    ]

]
from sklearn.metrics import classification_report, accuracy_score
best_models = []

best_results = []

for pipe, params in zip(models, params_grid):

    pipeline = Pipeline(pipe)

    

    gs = GridSearchCV(estimator=pipeline,

                      param_grid=params,

                      cv=3,

                      scoring='accuracy',

                      verbose=1,

                      n_jobs=-1)

    gs.fit(X_train, y_train)

    best_models.append(gs.best_estimator_)

    results = gs.cv_results_

    results['test_accuracy'] = accuracy_score(y_test, gs.best_estimator_.predict(X_test))

    best_results.append(pd.DataFrame(results))
pd.set_option('display.max_colwidth', 0)

display(pd.concat(best_results)[['rank_test_score', 'test_accuracy', 'mean_test_score',  'params']].sort_values('test_accuracy', ascending=False).head(20))