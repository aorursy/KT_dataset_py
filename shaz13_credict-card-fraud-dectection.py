# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (16,7)

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.shape
df.head(10)
plt.figure(figsize = (16,7))

sns.lmplot(x='Amount', y='Time', hue='Class', data=df)
plt.figure(figsize = (16,7))

plt.scatter(x= 'Time', y='Amount', data= df[df['Class']==1])

plt.xlabel('Time')

plt.ylabel('Amount')
X = df.drop('Class', axis=1)

y = df['Class']
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
print ("Train Score: {}".format(logmodel.score(X_train,y_train)))

print ("Test Score: {}".format(logmodel.score(X_test,y_test)))
scores = cross_val_score(logmodel, X, y) 

print("Cross-validation scores: {}".format(scores))
from sklearn.metrics import confusion_matrix,accuracy_score
print ('Confusion Matrix:')

print (confusion_matrix(y_test, y_pred))
print ("Accuracy Score:")

print (accuracy_score(y_test, y_pred))