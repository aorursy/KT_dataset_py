# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/titanic/train.csv')

df.describe()
df.describe(include=['O'])
X = df.drop('Survived', axis=1)

y = df[['Survived']]

X.shape, y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



y_train = np.ravel(y_train)

y_test = np.ravel(y_test)
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median'))])

    #('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

categorical_features = X_train.select_dtypes(include=['object']).columns
from sklearn.compose import ColumnTransformer



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', RandomForestClassifier())])
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf.score(X_test, y_test)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

   # KNeighborsClassifier(3),

   # SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    LogisticRegression(),

    #DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier()]



for classifier in classifiers:

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('classifier', classifier)])

    pipe.fit(X_train, y_train)   

    print(classifier)

    print("model score: %.3f" % pipe.score(X_test, y_test))
param_grid = { 

    'classifier__n_estimators': [200, 500, 700],

    'classifier__max_features': ['auto', 'sqrt', 'log2'],

    'classifier__max_depth' : [4,7,9,12],

    'classifier__criterion' :['gini', 'entropy']}



from sklearn.model_selection import GridSearchCV



CV = GridSearchCV(rf, param_grid, n_jobs= 1)

                  

CV.fit(X_train, y_train)  

print(CV.best_params_)    

print(CV.best_score_)
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

y_pred = rf.predict(test_df)



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": y_pred

    })

if submission.shape == (418,2):

    submission.to_csv('submission.csv', index=False)

    print('Submission ready')
submission.shape == (418,2)