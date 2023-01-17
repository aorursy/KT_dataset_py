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
data = pd.read_csv('../input/mushrooms.csv')
from sklearn.preprocessing import LabelEncoder

ch = list(data.columns.values)

for i in ch:

    encoder = LabelEncoder()

    col = data[i]

    col = encoder.fit_transform(col)

    data[i]=col
data.head()
x = data.drop('class', axis=1)

y = data['class']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =.3, random_state = 12)
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression().fit(x_train,y_train)

pred = clf.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred)
from sklearn.ensemble import RandomForestClassifier

clf2=RandomForestClassifier().fit(x_train,y_train)

pred2 = clf2.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred2)
from sklearn.tree import DecisionTreeClassifier

clf3=DecisionTreeClassifier().fit(x_train,y_train)

pred3 = clf3.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred3)
from sklearn.ensemble import GradientBoostingClassifier

clf4=GradientBoostingClassifier().fit(x_train,y_train)

pred4 = clf4.predict(x_test)

from sklearn.metrics import accuracy_score

accuracy_score(y_test,pred4)