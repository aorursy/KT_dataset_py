import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from patsy import dmatrices

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn import metrics



%matplotlib inline



df = pd.read_csv('../input/train.csv')

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import Imputer, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin



class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns

    

    def transform(self, X, *_):

        if isinstance(X, pd.DataFrame):

            return pd.DataFrame(X[self.columns])

        else:

            raise TypeError("This transformer only works with Pandas Dataframes")

    

    def fit(self, X, *_):

        return self

    

cs = ColumnSelector('Age')



cs.transform(df).head()



age_pipe = make_pipeline(ColumnSelector('Age'),

                         Imputer(),

                         StandardScaler())



df.Embarked = df.Embarked.fillna('S')



class GetDummiesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns):

        self.columns = columns

    

    def transform(self, X, *_):

        if isinstance(X, pd.DataFrame):

            return pd.get_dummies(X[self.columns], columns = self.columns)

        else:

            raise TypeError("This transformer only works with Pandas Dataframes")

    

    def fit(self, X, *_):

        return self

    

one_hot_pipe = GetDummiesTransformer(['Pclass', 'Embarked'])



class TrueFalseTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, flag):

        self.flag = flag

    

    def transform(self, X, *_):

        return X == self.flag



    def fit(self, X, *_):

        return self



gender_pipe = make_pipeline(ColumnSelector('Sex'),

                            TrueFalseTransformer('male'))



fare_pipe = make_pipeline(ColumnSelector('Fare'),

                          StandardScaler())



union = make_union(age_pipe,

                   one_hot_pipe,

                   gender_pipe,

                   fare_pipe)



X = df[[u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare', u'Embarked']]



X_1 = union.fit_transform(X)



new_cols = ['scaled_age', 'Pclass_1', 'Pclass_2', 'Pclass_3',

            'Embarked_C', 'Embarked_Q', 'Embarked_S',

            'male', 'scaled_fare']



Xt = pd.DataFrame(X_1, columns=new_cols)

Xt = pd.concat([Xt, X[[u'SibSp', u'Parch']]], axis = 1)
X = Xt

y = df[u'Survived']



from sklearn.feature_selection import RFECV

from sklearn.svm import SVR



estimator = SVR(kernel="linear")

selector = RFECV(estimator, step=1, cv=3)



rfecv_columns = selector.fit_transform(X,y)



rfecv_columns = Xt.columns[selector.support_]

X = X[rfecv_columns]
from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y)



from sklearn.grid_search import GridSearchCV



logreg_parameters = {

    'penalty':['l1','l2'],

    'C':np.logspace(-5,1,50),

    'solver':['liblinear']

}



lr = LogisticRegression(solver='liblinear')

mdl = lr.fit(X_train,y_train)



gs = GridSearchCV(lr, logreg_parameters, cv=5)



gs.fit(X_train,y_train)



predictions = gs.predict(X)
submission = pd.DataFrame({

        "PassengerId": df["PassengerId"],

        "Survived": predictions

    })



submission.to_csv('titanic.csv', index=False)