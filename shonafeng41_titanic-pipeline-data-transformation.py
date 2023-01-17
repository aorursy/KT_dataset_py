# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load dataset
def load_data(filename, datapath='../input'):
    path=os.path.join(datapath,filename)
    return pd.read_csv(path)
train= load_data('train.csv')
test = load_data('test.csv')
train.info()

train.head()
#numeric features
num_features=['Age','SibSp','Parch','Fare']
#categorical features
cat_features=['Pclass','Sex','Cabin','Embarked']

%matplotlib inline
import matplotlib.pyplot as plt
train[num_features].hist()
plt.show()
train[num_features].isnull().any()
train[cat_features].isnull().any()
train['Cabin'].value_counts().head()
# new feature - total family member on the ship
from sklearn.base import BaseEstimator, TransformerMixin
SibSp_ix, Parch_ix=1,2
class NewFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_total_family=True):
        self.add_total_family=add_total_family
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.add_total_family:
            total_family=X[:,SibSp_ix]+X[:,Parch_ix]
            return np.c_[X,total_family]
        else:
            return []

cabin_ix= 2     
class NewFeatureAdderCat(BaseEstimator, TransformerMixin):
    def __init__(self, add_cabin_letter=True):
        self.add_cabin_letter=add_cabin_letter
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['Cabin'].fillna('NaN',inplace=True)
        if self.add_cabin_letter:
            cabin_letter=X['Cabin'].apply(lambda X: X if X=='NaN' else X[0])
            X.drop('Cabin',inplace=True,axis=1)
            df=pd.DataFrame(np.c_[X,cabin_letter],columns=['Pclass','Sex','Embarked','Cabin'])
            return pd.get_dummies(df)
        else:
            return []
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
# creat feature transformation pipline for numeric features
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion



num_pipe=Pipeline([('selector',DataFrameSelector(num_features)),
                    ('imputer',Imputer(strategy='median')),
                  ('feature_adder',NewFeatureAdder()),
                  ('minmax_scaler',MinMaxScaler())])

# creat feature transformation pipline for categorical features

cat_pipe=Pipeline([('selector',DataFrameSelector(cat_features)),
                  ('feature_adder',NewFeatureAdderCat())
                  ])

# creat pipline for both numeric and categorical features
full_pipe = FeatureUnion(transformer_list=[
    ('num_pipe',num_pipe),
    ('cat_pipe',cat_pipe)
])
cat_feature_new=cat_pipe.fit_transform(train[cat_features])
cat_feature_new.shape
num_feature_new=num_pipe.fit_transform(train[num_features])
num_feature_new.shape
train_X=full_pipe.fit_transform(train)
test_X=full_pipe.fit_transform(test)
train_y=train['Survived']
# shuffle index
shuffle_index=np.random.permutation(len(train))
train_X, train_y=train_X[shuffle_index], train_y[shuffle_index]
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# target is balanced
train_y.value_counts()
def cross_val(classifier):
    model=classifier
    score=cross_val_score(model,train_X,train_y,cv=4,n_jobs=-1,scoring='accuracy')
    print('average accuracy: {} , std: {}'.format(np.mean(score),np.std(score)) )
cross_val(SGDClassifier(max_iter=5))
#selected for grid search
cross_val(LogisticRegression())
cross_val(RandomForestClassifier())
#selected for grid search
cross_val(KNeighborsClassifier())
from sklearn.model_selection import GridSearchCV

param ={'penalty':['l1','l2'],
       'C':[0.1, 0.5, 1, 5]}
clf=LogisticRegression()
best_log=GridSearchCV(clf,param,scoring='accuracy',cv=4,n_jobs=-1)
best_log.fit(train_X,train_y)
print('logistic regression accuracy: {} \n best parameters: {}'.format(best_log.best_score_,best_log.best_params_))

param={'n_neighbors':[3,5,10],
      'weights': ['uniform','distance'],
      'p':[1,2],
      'algorithm':['ball_tree','kd_tree'],
       'leaf_size':[10,30]
      }
clf=KNeighborsClassifier()
best_knn=GridSearchCV(clf,param,cv=4,n_jobs=-1,scoring='accuracy')
best_knn.fit(train_X,train_y)
print('KNN accuracy: {} \n best parameters: {}'.format(best.best_score_,best.best_params_))

# best model
best_clf=LogisticRegression(**best_log.best_params_)

