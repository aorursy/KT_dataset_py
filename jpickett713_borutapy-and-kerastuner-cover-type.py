# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, make_scorer

from sklearn.compose import ColumnTransformer



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install pdpipe

!pip install keras-tuner==1.0.0
from boruta import BorutaPy

import pdpipe as pdp

import tensorflow as tf

import kerastuner as kt
data=pd.read_csv('/kaggle/input/learn-together/train.csv.zip')

print(data.shape)
X=data.iloc[:,:55]

y=data.iloc[:,-1]



count_NA=[sum(X[col].isna()) for col in X.columns]

print(count_NA)
print(X.columns)

print(X.transpose()[11:55])
#encoding and scaling the variable with pdpipe

X_new=pdp.OneHotEncode().apply(X)

y_new=LabelEncoder().fit_transform(y)

X_new=pdp.Scale('StandardScaler').apply(X_new)

X_new=pdp.ColDrop('Id').apply(X_new)

print(X_new.shape)
X_test=pd.read_csv('/kaggle/input/learn-together/test.csv.zip')

print(X_test.shape)

X_test_final=pdp.OneHotEncode().apply(X_test)

X_test_final=pdp.Scale('StandardScaler').apply(X_test_final)

X_test_final=pdp.ColDrop('Id').apply(X_test_final)

print(X_test_final.shape)
# Using selectors to pluck out most important features

K_selector=SelectKBest(k=13)

K_selector_feat=K_selector.fit(X_new,y_new)

print(K_selector_feat.get_support())

X_k=K_selector.transform(X_new)





# using boruta to select features

X_bor=X_new.values

y_bor=y_new



rf= RandomForestClassifier()



feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X_bor,y_bor)



X_filtered=feat_selector.transform(X_bor)

print(X_filtered.shape)
# breaking the filtered datasets into train and validation sets

X_train, X_val, y_train, y_val=train_test_split(X_filtered,y_new,test_size=0.1)

X_k_train, X_k_val, y_K_train, y_k_val=train_test_split(X_k,y_new,test_size=0.1)



#building model to tune the hyperparameters

def build_model(hp):

  model_type = hp.Choice('model_type', ['random_forest', 'ridge', 'gbc'])

  if model_type == 'random_forest':

    model = RandomForestClassifier(

        n_estimators=hp.Int('n_estimators', 70, 120, step=10),

        max_depth=hp.Int('max_depth', 15, 25)

        )

  elif model_type== 'gbc':

      model = GradientBoostingClassifier(

          n_estimators=hp.Int('n_estimators', 90, 180, step=10),

          max_depth=hp.Int('max_depth', 1, 8),

          learning_rate=hp.Float('lr', 1e-3, 1, sampling='log')

          )

  else:

    model = RidgeClassifier(

        alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))

  return model



tuner = kt.tuners.Sklearn(

    oracle=kt.oracles.BayesianOptimization(

        objective=kt.Objective('score', 'max'),

        max_trials=10),

    hypermodel=build_model,

    scoring= make_scorer(accuracy_score),

    cv= StratifiedKFold(5),

    directory='.',

    project_name='my_proj')
# tuned on both boruta and Select K best data was better in this instance

tuner.search(X_k_train, y_K_train)

best_model = tuner.get_best_models(num_models=1)[0]

print(best_model)

pbest=best_model.predict(X_k_val)

print(accuracy_score(y_k_val,pbest))
m1=RandomForestClassifier(n_estimators=120, max_depth=19).fit(X_k_train,y_K_train)



p1=m1.predict(X_k_val)



print(accuracy_score(y_k_val,p1))

X_test_final=K_selector.transform(X_test_final)

test_preds=m1.predict(X_test_final)

my_submission = pd.DataFrame({'ID': X_test.Id, 'TARGET': test_preds})



my_submission.to_csv('submission.csv', index=False)