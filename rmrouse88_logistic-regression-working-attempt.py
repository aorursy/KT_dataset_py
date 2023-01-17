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
dframe = pd.read_csv("../input/creditcard.csv")
dframe.head()
len(dframe["Time"].unique())
X = dframe.ix[:,:-1]

X.head()
y = dframe.ix[:,-1]

y.head()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=0)
lr = LogisticRegression(C=1000.0, random_state=0)

lr.fit(X_train,y_train)
lr.coef_
fuck = lr.predict_proba(X_test)
count = 0

transactions=[]

for i in enumerate(fuck):

    if i[1][1]>i[1][0]:

        transactions.append(i[0])

        print(i[0])

        count += 1
print(count)

print(transactions)
X_test[transactions]