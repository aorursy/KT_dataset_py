import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline

import pandas as pd

path='../input/telecommunications/teleCust1000t.csv'

df = pd.read_csv(path)

df.head()
df['custcat'].value_counts()
len(df)
df.hist(column='income', bins=100)
df.columns
import numpy as np

X= df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values

X[0:3]
Y= df['custcat'].values

Y[0:3]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:3]
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.25, random_state=40)

print ('Train set:', X_train.shape,  Y_train.shape)

print ('Test set:', X_test.shape,  Y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)

neigh
Yp = neigh.predict(X_test)

Yp[0:3]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test,Yp))
# k=5

neigh5 = KNeighborsClassifier(n_neighbors = 5).fit(X_train,Y_train)

Yp5 = neigh5.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh5.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, Yp5))
# k=6

k=6

neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,Y_train)

yhat6 = neigh6.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(Y_train, neigh6.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(Y_test, yhat6))
Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,Y_train)

    Yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(Y_test, Yhat)



    

    std_acc[n-1]=np.std(Yhat==Y_test)/np.sqrt(Yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Neighbors (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 