# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from sklearn.linear_model import LogisticRegression


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#get info on the csv and if panda did parse it without trouble
train.info()
train.describe()
#remove cabin and ticket column
train = train.drop(['Cabin','Ticket'],axis=1)
test = test.drop(['Cabin','Ticket'],axis=1)
#removes all rows with NAN value
train = train.dropna()
test = test.fillna(0)
lr = LogisticRegression()

columns = ["Age"]
lr.fit(X = np.asarray(train[columns]),
       y = np.asarray(train.Survived).transpose())
prediction = lr.predict(np.asarray(test[columns]))
submission = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": prediction})
submission.to_csv('titanic.csv',index=False)
lr.score(np.asarray(train[columns]),np.asarray(train.Survived).transpose())
submission
