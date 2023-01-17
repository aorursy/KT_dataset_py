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

test = pd.read_csv('../input/test.csv')
from sklearn.model_selection import train_test_split

from sklearn.linear_model.logistic import LogisticRegression

from sklearn.svm import SVC
images = train.iloc[0:5000,1:]

labels = train.iloc[0:5000,:1]

X_train, X_test,y_train, y_test = train_test_split(images, labels, train_size=0.8, random_state=0)
#splitting data in 75-25 ration

#X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.3)
#X_test.head()
#fitting logistic regression model

logreg = LogisticRegression()

logreg.fit(X_train,y_train.values.ravel())

logreg.score(X_test, y_test)
X_train[X_train>0] = 1

X_test[X_test>0] = 1
# svm

sv = SVC()

sv.fit(X_train, y_train.values.ravel())

sv.score(X_test, y_test)
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=500,learning_rate=0.1, max_depth=4)

gbc.fit(X_train, y_train.values.ravel())

gbc.score(X_test, y_test)
Y_pred = gbc.predict(test)
df = pd.DataFrame(Y_pred)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('submission.csv',header=True)