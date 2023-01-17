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




import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score



from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
## Backup Imports

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import QuantileTransformer



from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

import catboost as cb

import lightgbm as lgb



from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score



from mlxtend.feature_selection import ColumnSelector

from sklearn.pipeline import make_pipeline
train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")
# Remove the Labels and make them y

y = train['Cover_Type']



# Remove label from Train set

X = train.drop(['Cover_Type'],axis=1)



# Rename test to text_X

test_X = test







# split data into training and validation data

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)



X = X.drop(['Id'], axis = 1)

train_X = train_X.drop(['Id'], axis = 1)

val_X = val_X.drop(['Id'], axis = 1)

test_X = test_X.drop(['Id'], axis = 1)
train_X.describe()
val_X.describe()
test_X.describe()


rfcfin = RandomForestClassifier(n_estimators = int(1631.3630739649345),

                                min_samples_split = int(2.4671165024828747),

                                min_samples_leaf = int(1.4052032266878376),

                                max_features = 0.23657708614689418,

                                max_depth = int(426.8410655510125),

                                bootstrap = int(0.8070235824535138),

                                random_state=42)

rfcfin.fit(X, y.ravel())
test_ids = test["Id"]

test_pred = rfcfin.predict(test_X.values)
# Save test predictions to file

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)