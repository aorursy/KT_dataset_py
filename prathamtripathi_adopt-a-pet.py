import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pylab as py

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
!pip install pydotplus
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree
import collections

import pydotplus

import matplotlib.image as mpimg
df = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/train.csv")

df.head()
test_data = pd.read_csv("../input/hackerearth-ml-challenge-pet-adoption/test.csv")

test_data.head()
df = df.dropna()

df.head()
df['breed_category'].value_counts()
X = df[['length(m)', 'height(cm)',"X1","X2"]]

X[0:5]
y = df["pet_category"]

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 123)
print("Training Set: ",X_train.shape,y_train.shape)

print("Testing Set: ",X_test.shape,y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

k = 9

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh
y_hat = neigh.predict(X_test)

y_hat[0:5]

np.unique(y_hat)
from sklearn import metrics

print("Accuracy Score is : ", metrics.accuracy_score(y_test,y_hat))
ks = 10

mean_acc = np.zeros((ks - 1))

std_acc = np.zeros((ks - 1))

confusion_matrix = []



for n in range(1,ks):

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat = neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)

    std_acc[n-1] = np.std(yhat == y_test)/np.sqrt(yhat.shape[0])

    

mean_acc
plt.plot(range(1,ks),mean_acc,'g')

plt.fill_between(range(1,ks),mean_acc - 1*std_acc,mean_acc + 1*std_acc,alpha = 1)

plt.legend(["Accuracy", "+/- 3xstd"])

plt.tight_layout()

print("The best accuracy for the model is :",mean_acc.max(),"With k=",mean_acc.argmax()+1)

plt.show()
PetTree = DecisionTreeClassifier(criterion ="entropy", max_depth = 5)

PetTree.fit(X_train,y_train)
yhat1 = PetTree.predict(X_test) 

print(yhat1[0:5])

print(y_test[0:5])
print("Accuracy Score is : ", metrics.accuracy_score(y_test,yhat1))
Xtest = test_data[['length(m)', 'height(cm)', 'X1', 'X2']]

Xtest.head()
pred = PetTree.predict(Xtest)
data_feature_names = ['length(m)', 'height(cm)', 'X1', 'X2']
# Visualize data

dot_data = tree.export_graphviz(PetTree,

                                feature_names=data_feature_names,

                                out_file=None,

                                filled=True,

                                rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)



colors = ('turquoise', 'orange')

edges = collections.defaultdict(list)



for edge in graph.get_edge_list():

    edges[edge.get_source()].append(int(edge.get_destination()))



for edge in edges:

    edges[edge].sort()    

    for i in range(2):

        dest = graph.get_node(str(edges[edge][i]))[0]

        dest.set_fillcolor(colors[i])

filename = "tree.png"

graph.write_png(filename)

img = mpimg.imread(filename)

plt.figure(figsize=(100,200))

plt.imshow(img,interpolation = 'nearest')

plt.show()
output = pd.DataFrame({'PetId': test_data.pet_id, 'Pet Category': pred})

output.to_csv('Output.csv', index=False)

print("Your submission was successfully saved!")