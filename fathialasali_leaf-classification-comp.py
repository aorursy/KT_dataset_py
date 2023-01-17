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
import pandas as pd

import numpy as np

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.model_selection import cross_val_score

from sklearn.metrics import log_loss

from sklearn.preprocessing import LabelBinarizer

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('../input/leaf-classification/train.csv.zip' , index_col = False)

train
x_train = train.drop(['id', 'species'], axis=1).values
train
le = LabelEncoder().fit(train['species']) 

y_train = le.transform(train['species'])
y_train
scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial') #Simple Logistic Regression

clf.fit(x_train, y_train) #Fit the model
test = pd.read_csv('../input/leaf-classification/test.csv.zip' ,index_col = False )

test
test_ids = test.pop('id') 

x_test = test.values 
x_test = test.values 
x_test = scaler.transform(x_test)

y_test = clf.predict_proba(x_test)
submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.head(2)
submission.to_csv('./submission_leaf_classification.csv')

print('Done')