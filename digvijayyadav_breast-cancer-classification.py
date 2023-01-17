import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

import seaborn as sns

from sklearn import preprocessing

%matplotlib inline
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
df['radius_mean'].value_counts()



print(df.shape)

def diagnosis_value(diagnosis): 

    if diagnosis == 'M': 

        return 1

    else: 

        return 0

  

    df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

    df.drop(['Unnamed: 32','id'],axis=1)
plt.hist(df['radius_mean'])
X = df[[ 'radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']] .values  #.astype(float)

X[0:4]

y= np.array(df['diagnosis'])
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

print(neigh)
yhat = neigh.predict(X_test)

yhat[0:5]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
# write your code here

k=6

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

print(neigh)

#predicting

yhat6=neigh.predict(X_test)

print('Train Accuracy is : ',metrics.accuracy_score(y_train, neigh.predict(X_train)))

print('test accuracy is : ', metrics.accuracy_score(y_test, yhat6))

Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of estimators (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 