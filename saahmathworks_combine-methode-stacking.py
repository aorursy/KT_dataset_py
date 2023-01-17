from datetime import datetime



print("last update: {}".format(datetime.now())) 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,  GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier,  ExtraTreesClassifier

import lightgbm as lgb

from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, roc_auc_score

from sklearn.svm import SVC 

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMModel,LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

import numpy as np

np.random.seed(0)
# Read the data

X_original = pd.read_csv('/kaggle/input/learn-together/train.csv', index_col='Id')

X_test = pd.read_csv('/kaggle/input/learn-together/test.csv', index_col='Id')

X = X_original.drop('Cover_Type', axis = 1)

y = X_original['Cover_Type']
X.head()
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from xgboost import XGBClassifier



from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.multiclass import OneVsRestClassifier



Experiments = {"Algo":["RandomForestClassifier", "LGBMClassifier", "KNeighborsClassifier", 'XGBClassifier'],

              "object": [lambda: OneVsRestClassifier(RandomForestClassifier(n_estimators = 1000, max_features = 'sqrt')),

                        lambda: OneVsRestClassifier(LGBMClassifier(learning_rate =0.05, n_estimators=1000)),

                        lambda: OneVsRestClassifier(KNeighborsClassifier()),

                        #lambda: SVC(gamma='scale', kernel='rbf', probability=True),

                        lambda: OneVsRestClassifier(XGBClassifier(learning_rate =0.05, n_estimators=1000))],

               "prediction": [[] for _ in range(5)]}





scale = StandardScaler()

#X_copy.iloc[:,0:10] = scale.fit_transform(X.iloc[:,0:10])

[_.shape for _ in train_test_split(X, y, test_size = 0.5)]
X[X['Aspect']==0]
X[X['Horizontal_Distance_To_Hydrology'] == 0]
# O stand for missing values

dict = {}

for col in X.columns.tolist()[:10]:

    dict[col] = X[X[col] == 0].shape[0]

dict    
# explicitly require this experimental feature

from sklearn.experimental import enable_iterative_imputer  # noqa

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

# regressor to use for imputing

from sklearn.ensemble import ExtraTreesRegressor



#estimator = ExtraTreesRegressor(n_estimators=10, random_state=0) estimator = estimator, 

imputer = IterativeImputer(missing_values = 0, max_iter=50, random_state=0)

X_mat = imputer.fit_transform(X.iloc[:,:10].values)

 
X_test_mat = imputer.transform(X_test.iloc[:,:10].values)
X_mat.shape
X_cat = X.iloc[:,10:]

X_test_cat = X_test.iloc[:,10:]
train_X = np.hstack((X_mat, X_cat.values))

test_X = np.hstack((X_test_mat, X_test_cat.values))
train_X_df = pd.DataFrame(train_X, columns = X.columns.tolist(), index = X.index)

test_X_df = pd.DataFrame(test_X, columns = X_test.columns.tolist(), index = X_test.index)

train_X_df.head()
X.head()
test_X_df.head()
# O stand for missing values

dict2 = {}

for col in train_X_df.columns.tolist()[:10]:

    dict2[col] = train_X_df[train_X_df[col] == 0].shape[0]

dict2  
# Meta Classifier

meta_cls = XGBClassifier(learning_rate =0.1, n_estimators=500)
list_estimators = [RandomForestClassifier(n_estimators=400, max_features = 'sqrt',

                                random_state=1, n_jobs=-1), 

                   XGBClassifier(learning_rate =0.1, n_estimators=400, random_state=1, n_jobs=-1), 

                   LGBMClassifier(n_estimators=400,verbosity=0, random_state=1, n_jobs=-1),

                   KNeighborsClassifier(n_jobs = -1)]

base_methods = list(zip(Experiments["Algo"], list_estimators))

#base_methods 
y.values
train_X
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(train_X, y.values, test_size = 0.15, random_state=0)
from mlxtend.classifier import StackingCVClassifier



state = 1

stack = StackingCVClassifier(classifiers=list_estimators,

                             meta_classifier=meta_cls,

                             cv=3,

                             use_probas=True,

                             verbose=1, 

                             random_state=state,

                             n_jobs=-1)

stack = stack.fit(X_train,y_train)
X_valid
y_val_pred = stack.predict(X_valid)
#performances

y_vp_val = stack.predict(X_valid)

y_vp_train = stack.predict(X_train)

print('f1_score', f1_score(y_valid, y_vp_val, average='weighted'))

print('acc_score_train', accuracy_score(y_train, y_vp_train))

print('acc_score_valid', accuracy_score(y_valid, y_vp_val))
preds_test = stack.predict(test_X)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'Cover_Type': preds_test})

output.to_csv('submission.csv', index=False)