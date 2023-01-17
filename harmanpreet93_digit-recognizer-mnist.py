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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train_label = train['label']
train_label.head()
train.drop('label',axis=1, inplace=True)

train.head()
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier
decisionTreeClf = DecisionTreeClassifier(max_depth=5)

adaBoostClf = AdaBoostClassifier(decisionTreeClf, n_estimators=50,learning_rate=1)

adaBoostClf.fit(train, train_label)

result = adaBoostClf.predict(test)
sub = pd.DataFrame({'Label': result}, index=range(1,len(result)+1))

sub.to_csv('submission_adaboost.csv',index_label='ImageId')
sub1 = pd.read_csv('submission_adaboost.csv')

sub1.head()
ls