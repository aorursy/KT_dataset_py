import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/column_2C_weka.csv')
data.head()
data.info()
data.describe()
sns.pairplot(data,hue = 'class',height = 3)
plt.figure(figsize = (10,6))

sns.heatmap(data.corr(),alpha = 0.8,annot = True)
sns.jointplot(x = 'pelvic_incidence',y = 'pelvic_radius',data = data,color = 'green')
A = data[data['class'] == 'Abnormal']

N = data[data['class'] == 'Normal']
plt.scatter(A.pelvic_incidence, A.pelvic_radius, color = 'purple', label = 'Abnormal',alpha = 0.7)

plt.scatter(N.pelvic_incidence, N.pelvic_radius, color = 'orange', label = 'Normal', alpha = 0.5)

plt.xlabel('pelvic_incidence')

plt.ylabel('pelvic_radius')

plt.legend()

plt.show()
CLASS = pd.get_dummies(data['class'],drop_first = True)
data = pd.concat([data,CLASS],axis = 1)
data.head()
data.drop(['class'],axis = 1,inplace = True)
data.head()
y = data['Normal'].values

x_data = data.drop(['Normal'], axis=1)
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)
#knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 21)

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print('{} KNN score: {}'.format(21,knn.score(x_test,y_test)))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,p = 2,metric = 'euclidean')
knn.fit(x,y)
pred = knn.predict(x)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y,pred))

print(classification_report(y,pred))
score_list = []

for each in range(1,22):

    knn2 = KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x, y)

    score_list.append(knn2.score(x,y))

    

plt.figure(figsize = (10,6))

plt.plot(range(1,22),score_list,color = 'blue',linestyle = 'dashed',marker = 'o',markerfacecolor = 'gold',markersize = 10)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
# Model complexity

neig = np.arange(1, 22)

train_accuracy = []

test_accuracy = []

# Loop over different values of k

for i, k in enumerate(neig):

    # k from 1 to 25(exclude)

    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit with knn

    knn.fit(x_train,y_train)

    #train accuracy

    train_accuracy.append(knn.score(x_train, y_train))

    # test accuracy

    test_accuracy.append(knn.score(x_test, y_test))



# Plot

plt.figure(figsize=[13,8])

plt.plot(neig, test_accuracy, label = 'Testing Accuracy',color = 'blue',linestyle = 'dashed',marker = 'o',markerfacecolor = 'gold',markersize = 10)

plt.plot(neig, train_accuracy, label = 'Training Accuracy',color = 'teal',linestyle = '-',marker = 'o',markerfacecolor = 'red',markersize = 10)

plt.legend()

plt.title('-value VS Accuracy')

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.xticks(neig)

plt.savefig('graph.png')

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))