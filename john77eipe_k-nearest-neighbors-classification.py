# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from collections import Counter

import math



def knn(data, query, k, distance_fn, choice_fn):

    neighbor_distances_and_indices = []

    

    # 3. For each example in the data

    for index, example in enumerate(data):

        # 3.1 Calculate the distance between the query example and the current

        # example from the data.

        distance = distance_fn(example[:-1], query)

        

        # 3.2 Add the distance and the index of the example to an ordered collection

        neighbor_distances_and_indices.append((distance, index))

    

    # 4. Sort the ordered collection of distances and indices from

    # smallest to largest (in ascending order) by the distances

    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    

    # 5. Pick the first K entries from the sorted collection

    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    

    # 6. Get the labels of the selected K entries

    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]



    # 7. If regression (choice_fn = mean), return the average of the K labels

    # 8. If classification (choice_fn = mode), return the mode of the K labels

    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)



def mean(labels):

    return sum(labels) / len(labels)



def mode(labels):

    return Counter(labels).most_common(1)[0][0]



def euclidean_distance(point1, point2):

    sum_squared_distance = 0

    for i in range(len(point1)):

        sum_squared_distance += math.pow(point1[i] - point2[i], 2)

    return math.sqrt(sum_squared_distance)





'''

# Regression Data

# 

# Column 0: height (inches)

# Column 1: weight (pounds)

'''

reg_data = [

   [65.75, 112.99],

   [71.52, 136.49],

   [69.40, 153.03],

   [68.22, 142.34],

   [67.79, 144.30],

   [68.70, 123.30],

   [69.80, 141.49],

   [70.01, 136.46],

   [67.90, 112.37],

   [66.49, 127.45],

]



# Question:

# Given the data we have, what's the best-guess at someone's weight if they are 60 inches tall?

reg_query = [60]

reg_k_nearest_neighbors, reg_prediction = knn(

    reg_data, reg_query, k=3, distance_fn=euclidean_distance, choice_fn=mean

)



'''

# Classification Data

# 

# Column 0: age

# Column 1: likes pineapple

'''

clf_data = [

   [22, 1],

   [23, 1],

   [21, 1],

   [18, 1],

   [19, 1],

   [25, 0],

   [27, 0],

   [29, 0],

   [31, 0],

   [45, 0],

]

# Question:

# Given the data we have, does a 33 year old like pineapples on their pizza?

clf_query = [33]

clf_k_nearest_neighbors, clf_prediction = knn(

    clf_data, clf_query, k=3, distance_fn=euclidean_distance, choice_fn=mode

)

import seaborn as sns

from sklearn import preprocessing



df = pd.read_csv('/kaggle/input/teleCust1000t.csv')

df.head()
df['custcat'].value_counts()
sns.distplot(df["income"], kde=False, bins=100)
df.columns
#Converting to numpy array

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)

X[0:5]

#Converting the labels also

y = df['custcat'].values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
k = 4

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
yhat = neigh.predict(X_test)

yhat[0:5]
from sklearn import metrics

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))

print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
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
import matplotlib.pyplot as plt

plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)

plt.legend(('Accuracy ', '+/- 3xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 