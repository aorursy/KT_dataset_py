import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score,confusion_matrix

import xgboost as xgb

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X_set = pd.read_csv('/kaggle/input/hackstat2k19/Trainset.csv')
X_test = pd.read_csv('/kaggle/input/hackstat2k19/xtest.csv')
y=X_set.Revenue
X_set.dropna(axis=0, subset=['Revenue'], inplace=True)
X_set.drop(['Revenue'], axis=1, inplace=True)
X_train, X_valid, y_train, y_valid = train_test_split(X_set, y,test_size=0.2,random_state=0)
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)
print (object_cols)
good_label_cols = [col for col in object_cols if 

                   set(X_train[col]) == set(X_valid[col])]
label_encoder = LabelEncoder()
for col in set(good_label_cols):

    X_train[col] = label_encoder.fit_transform(X_train[col])

    X_valid[col] = label_encoder.transform(X_valid[col])

    X_test[col] = label_encoder.transform(X_test[col])
X_set.head()
myimputer = SimpleImputer()
imX_train = pd.DataFrame(myimputer.fit_transform(X_train))

imX_valid = pd.DataFrame(myimputer.transform(X_valid))
imX_train.columns = X_train.columns
imX_valid.columns = X_valid.columns
model  = RandomForestClassifier(n_estimators=100,random_state=2)

#model = xgb.XGBClassifier(n_estimators=100)
X_train = imX_train

X_valid = imX_valid
model.fit(X_train,y_train)
pred = model.predict(X_valid)
error = accuracy_score(pred,y_valid)
print (error)
print(confusion_matrix(pred,y_valid))
column_with_missing = [col for col in X_test.columns

                      if X_test[col].isnull().any()]
print (column_with_missing)
X_test.dropna(axis=0, subset=['ID'], inplace=True)
X_test.drop(['ID'], axis=1, inplace=True)
X_test.head()
preds_test = model.predict(X_test)
output = pd.DataFrame({'ID': X_test.index+1,'Revenue': preds_test})

X_test.to_csv('new_submission.csv', index=None,header=True)
def newly(value):

    if value <=0.14:

        return 1

    else:

        return 0
X_set['exit']= X_set['ExitRates'].apply(newly)

X_test['exit']= X_set['ExitRates'].apply(newly)

X_valid.head()