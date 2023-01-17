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
#import tinhing data

data=pd.read_csv("../input/creditcard.csv")

#data

#data.to_csv('example_file.csv')
from sklearn.model_selection import train_test_split
#data percent of majority class

sum(data.Class==0)/len(data.Class)
#spearating  data into features and labels

X=data.iloc[:,:-1]

y=data['Class']

#print(X.head(),y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(C=1,).fit(X_train,y_train)

predictions=lr.predict(X_test)



lr.score(X_test,y_test)