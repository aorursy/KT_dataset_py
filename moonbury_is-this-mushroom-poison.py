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

data.describe()
y=data['class']

cols = set(data.columns)

cols.remove('class')

X=data[list(cols)]

print("shape of X and y",X.shape,y.shape)



from collections import Counter

#print(Counter(X['cap-shape']))

print(Counter(y))

import matplotlib.pyplot as plt

plt.figure(figsize=(8,3))

plt.title('Poison?')

c = pd.value_counts(data['class'],sort=True).sort_index()

c.plot(kind='bar')

X.head()
plt.figure(figsize=(20,40))



j=1

for i in cols:

    plt.subplot(11,2,j)

    j += 1

    c = pd.value_counts(data[i],sort=True).sort_index()

    c.plot(kind='bar')

    plt.title(i)
print("split the data to train set and test set. Since it is categorical data, use one hot encoding.")

from sklearn.cross_validation import train_test_split

one_hot_encoded_X_train = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(one_hot_encoded_X_train,y,train_size=0.8,random_state=101)

print("shape of one hot encoeded data, the features is 117 now.", X_train.shape, y_train.shape)
X_train.head()
print("run the data in logistic regression")

from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix

from sklearn.linear_model import LogisticRegression



regr = LogisticRegression()

regr.fit(X_train,y_train)

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)

print("training data accuracy: ",accuracy_score(y_train,y_train_pred))

print("testing data accuracy: ",accuracy_score(y_test,y_test_pred))



print("Classification Report")

print(classification_report(y_test,y_test_pred))



print("Confusion Matrix")

print(confusion_matrix(y_test,y_test_pred))