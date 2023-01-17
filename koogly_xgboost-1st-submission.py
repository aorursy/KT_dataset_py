



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/learn-together/train.csv", index_col='Id')

test = pd.read_csv("../input/learn-together/test.csv", index_col='Id')

X=train.drop(['Cover_Type'], axis=1)

X_test = test.copy()

y=train['Cover_Type'] # target

TARGET = 'Cover_Type'
X.head()
X.describe()
X.shape

print(X.isnull().any().any())

print(y.isnull().any().any())

print(X_test.isnull().any().any())



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score,cross_validate

X_train,  X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=1)

print(X_train.shape)

print(y_train.shape)

print(X_val.shape)

print(y_val.shape)
# fit model no training data

from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=200)

model.fit(X_train, y_train, 

             early_stopping_rounds=6, 

             eval_set=[(X_val, y_val)],

             verbose=False)
y_pred = model.predict(X_val)

predictions = [round(value) for value in y_pred]
from sklearn.metrics import mean_absolute_error



predictions = model.predict(X_val)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_val)))

from sklearn.metrics import accuracy_score



accuracy = accuracy_score(y_val, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.ensemble import ExtraTreesClassifier

SEED = 2007

model_2=ExtraTreesClassifier( max_depth=400, 

           n_estimators=450, n_jobs=-1,

           oob_score=False, random_state=SEED, 

           warm_start=True)

model_2.fit(X_train, y_train)

predictions_2 = model_2.predict(X_val)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions_2, y_val)))
accuracy = accuracy_score(y_val, predictions_2)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
test_pred = model_2.predict(test)
output = pd.DataFrame({'ID': test.index,

                       TARGET: test_pred })

output.to_csv('submission.csv', index=False)


