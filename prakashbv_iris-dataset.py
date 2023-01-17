# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
#Import Dataset from Seaborn library

iris = pd.read_csv("../input/IRIS.csv")
#checking the data

iris.head(10)
#Visualizing data based on the different features

sns.pairplot(data=iris, hue='species', palette='Set2')
#converting target variable classes into numerical

iris['species'] = iris['species'].replace('Iris-setosa',0)

iris['species'] = iris['species'].replace('Iris-versicolor',1)

iris['species'] = iris['species'].replace('Iris-virginica',2)
iris['species'].head(10)
#Splitting the dataset into train and test for model building

from sklearn.model_selection import train_test_split



x = iris.iloc[:,:-1]

y = iris.iloc[:,4]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30)
y.head()
#importing libraries for model 

from sklearn.svm import SVC
#Building model on train data

#model = SVC(kernel='rbf', gamma=0.7, C=1.0)

model =  SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,

    probability=False, random_state=None, shrinking=True, tol=0.001,

    verbose=False)

model.fit(x_train, y_train)
#Predicting for the test data

predictions = model.predict(x_test)
#importing libraries for model evaluation

from sklearn.metrics import classification_report, accuracy_score
#Confusion matrix

print(pd.crosstab(y_test, predictions))
print(classification_report(y_test, predictions))
#Checking Accuracy using accuracy_score

accuracy_score(y_test, predictions)*100
model2 = SVC(kernel='linear', C=1.0).fit(x_train, y_train)

predictions2 = model2.predict(x_test)

print(pd.crosstab(y_test, predictions2))

print(classification_report(y_test, predictions2))

model3 = SVC(kernel='poly', degree=3, C=1.0).fit(x_train, y_train)

predictions3 = model3.predict(x_test)

print(pd.crosstab(y_test, predictions3))

print(classification_report(y_test, predictions3))
model3.get_params
from sklearn.model_selection import GridSearchCV

# Grid Search

# Parameter Grid

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}

 

# Make grid search classifier

clf_grid = GridSearchCV(SVC(), param_grid, verbose=1)

 

# Train the classifier

clf_grid.fit(x_train, y_train)

 

# clf = grid.best_estimator_()

print("Best Parameters:\n", clf_grid.best_params_)

print("Best Estimators:\n", clf_grid.best_estimator_)
from mlxtend.plotting import plot_decision_regions

#from sklearn import datasets

#iris = datasets.load_iris()

X = np.array(iris.iloc[:, [0, 2]])

y = np.array(iris.iloc[:,4])



# Training a classifier

model_plot = SVC(C=1.0, kernel='linear')

model_plot.fit(X, y)





# Plotting decision regions

plot_decision_regions(X, y, clf=model_plot, legend=2)



# Adding axes annotations

plt.xlabel('sepal length [cm]')

plt.ylabel('petal length [cm]')

plt.title('SVM on Iris')

plt.show()
y_train.shape
from xgboost import XGBClassifier

clf = XGBClassifier()

clf

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
#importing library and segregation of data as train and test using DMatrix Data structure

import xgboost as xgb



dtrain = xgb.DMatrix(x_train, label=y_train)

dtest = xgb.DMatrix(x_test, label=y_test)
#paramaters 

param = {

    'max_depth': 3,  # the maximum depth of each tree

    'eta': 0.3,  # the training step for each iteration

    'silent': 1,  # logging mode - quiet

    'objective': 'multi:softprob',  # error evaluation for multiclass training

    'num_class': 3}  # the number of classes that exist in this datset

num_round = 5  # the number of training iterations





#model builing using training data

bst = xgb.train(param, dtrain, num_round)

pred = bst.predict(dtest)



#Check the Accuracy

bst.dump_model('dump.raw.txt')

#Calculating prediction accuracy

import numpy as np

best_preds = np.asarray([np.argmax(line) for line in pred])

from sklearn.metrics import precision_score

print(precision_score(y_test, best_preds, average='macro'))

# >> 1.0
# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



xgb.plot_tree(bst, num_trees=1)

fig = plt.gcf()

fig.set_size_inches(150, 100)

fig.savefig('treeIris.png') 
#Feature importance

from xgboost import plot_importance

from matplotlib import pyplot

plot_importance(bst)

pyplot.show()
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(x_train,y_train)

pred = clf.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 3)

classifier = KNeighborsClassifier(n_neighbors=3)



# Fitting the model

classifier.fit(x_train, y_train)



pred = classifier.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
x = iris.iloc[:, [1, 2, 3, 4]].values
x.shape


#Finding the optimum number of clusters for k-means classification

from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)

    

#Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') #within cluster sum of squares

plt.show()

#Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)
#Visualising the clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-versicolour')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')



#Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')



plt.legend()


print(pd.crosstab(iris['species'], kmeans.labels_))

print(classification_report(iris['species'], kmeans.labels_))