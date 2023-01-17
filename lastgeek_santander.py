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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.preprocessing import LabelEncoder, StandardScaler

import numpy as np
data = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/train.csv')
data.head()
x = data.drop(columns=['ID_code','target'])

x.head()
y = data.target
encode = LabelEncoder()

y = encode.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
scale = StandardScaler()

X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)
model = LogisticRegression(random_state=4)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
accuracy_score(predictions,y_test)
test = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/test.csv')

test.head()
test.drop(columns='ID_code',inplace=True)
test.head()
pred = model.predict(test)
print(predictions)
submission = pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv')
submission.head()
import xgboost as xg
boost = xg.XGBClassifier(obejective='reg:linear', objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
boost.fit(X_train,y_train)
pred = model.predict(X_test)
accuracy_score(pred,y_test)