# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
train_x = train[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]

train_y = train['type']
set(train['color'])
import pandas as pd

x = pd.get_dummies(train['color'])

#train_x = train_x.append(x)

train_features = pd.concat([train_x, x], axis=1)
train_features.head()
from sklearn.tree import DecisionTreeClassifier

cls = DecisionTreeClassifier()
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_features, train_y, test_size = 0.2, random_state = 0)
cls.fit(X_train, Y_train)
pred = cls.predict(X_test)
import numpy as np

from sklearn.metrics import fbeta_score

score = fbeta_score(np.array(Y_test), pred, beta=0.5, average='macro')
score
from sklearn.tree import DecisionTreeClassifier

cls1 = DecisionTreeClassifier()
test_x = pd.read_csv('../input/test.csv')

test_x.head()
import pandas as pd

y = pd.get_dummies(test_x['color'])

test_columns = test_x[["bone_length", "rotting_flesh", "hair_length", "has_soul"]]

test_features = pd.concat([test_columns, y], axis=1)
test_features.head()
cls1.fit(train_features, train_y)

pred_all = cls1.predict(test_features)
pred_all[0]
sub = test_x['id']
sub.head()
predictions = pd.DataFrame(pred_all)
sub = pd.concat([sub, predictions], axis=1)
sub.columns = ['id', 'type']
sub.head()
sub.to_csv('submit.csv', index=False)