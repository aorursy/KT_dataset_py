from IPython.display import Image

Image("KNN-Algorithm-using-Python-1.png")
from IPython.display import Image

Image("class_prediction.jpg")
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# read csv (comma separated value) into data

data = pd.read_csv('column_2C_weka.csv')

print(plt.style.available) # look at available plot styles

plt.style.use('_classic_test')

data.head()
data.shape
print('No of columns in the dataset:',data.columns.size)

print("Name of Columns:\n",data.columns.values)
data.info()
data.describe()
sns.pairplot(data,hue="class",palette="Set2")

plt.show()
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
data.loc[:,'class'].value_counts()
sns.countplot(x="class", data=data)

# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x,y)

prediction = knn.predict(x)

print('Prediction: {}'.format(prediction))
# train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 3)

x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))

print('With KNN (K=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy
# Model complexity

neig = np.arange(1, 25)

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

plt.plot(neig, test_accuracy, label = 'Testing Accuracy')

plt.plot(neig, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.title('Value VS Accuracy',fontsize=20)

plt.xlabel('Number of Neighbors',fontsize=20)

plt.ylabel('Accuracy',fontsize=20)

plt.xticks(neig)

plt.grid()

plt.show()

print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))
# grid search cross validation with 1 hyperparameter

from sklearn.model_selection import GridSearchCV

grid = {'n_neighbors': np.arange(1,25)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x,y)# Fit



# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))

from sklearn.neighbors import KNeighborsClassifier

grid = {'n_neighbors': np.arange(1,25)}

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV

knn_cv.fit(x_train, y_train)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn_cv.score(x_train, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn_cv.score(x_test, y_test)))

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 

print("Best score: {}".format(knn_cv.best_score_))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=12, p=6, metric='minkowski')

knn.fit(x_train, y_train)



print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))

print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))

from sklearn.neighbors import KNeighborsClassifier

train_score_knn=[]

test_score_knn=[]

for i in range(1,10):

    knn = KNeighborsClassifier(n_neighbors=12, p=i, metric='minkowski')

    knn.fit(x_train, y_train)

    train_score_knn.append(knn.score(x_train, y_train))

    test_score_knn.append(knn.score(x_test, y_test))

    print('For p value=',i)

    print('The accuracy of the knn classifier is {:.2f} out of 1 on training data'.format(knn.score(x_train, y_train)))

    print('The accuracy of the knn classifier is {:.2f} out of 1 on test data'.format(knn.score(x_test, y_test)))

    print()
