import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt 

from sklearn import datasets, linear_model



#%matplotlib inline



import seaborn as sns

df = pd.read_csv("../input/winedata/winequality_red.csv")
df.head(3)
df.isnull().any()
df.describe()
df.dtypes
sns.countplot(x="quality", data=df)
reviews = []

for i in df['quality']:

    if i >= 1 and i <= 5:

        reviews.append('0')

    elif i >= 6 and i <= 10:

        reviews.append('1')

df['reviews'] = reviews
sns.pairplot(df, vars=df.columns[:-1])
fig, ax = plt.subplots(figsize=(10,10))

corr = df.corr()



# plot the heatmap

sns.heatmap(corr,annot=True,

        xticklabels=corr.columns,

        yticklabels=corr.columns)
# Get from the dataframe (the independent variables)



X = df[['fixed acidity','citric acid','residual sugar','chlorides'

        ,'free sulfur dioxide','total sulfur dioxide','density', 'pH','sulphates','alcohol']] 



# Get from the dataframe the just created label variable (dependent variables)

y = df['reviews']
from sklearn.feature_selection import SelectKBest,f_classif



print(X.shape)

# Find K  best features   8 is good

kbest = SelectKBest(f_classif, k=8)

kbest = kbest.fit(X,y)

kbest.transform(X).shape

print(kbest.scores_)

print(kbest.transform(X).shape)

X.columns[kbest.get_support(indices=True)]

vector_names = list(X.columns[kbest.get_support(indices=True)])

print(vector_names)

X = df[['fixed acidity', 'citric acid', 'chlorides', 'free sulfur dioxide'

        , 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']]
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=33)
from sklearn.preprocessing import StandardScaler

#Standarize the features

#create a copy of the dataset

scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X_train[:5]
import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    print('Confusion matrix')

    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()

from sklearn import metrics

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    

    clf.fit(X_train, y_train)

    

    print("Accuracy on training set:")

    print(clf.score(X_train, y_train))

    print("Accuracy on testing set:")

    print(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    

    print("Classification Report:")

    print(metrics.classification_report(y_test, y_pred))

    

    # Compute confusion matrix

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    np.set_printoptions(precision=2)

    # Plot confusion matrix

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=[0,1],

                      title='Confusion matrix')



    plt.show()
import numpy as np

import matplotlib.pyplot as plt

from sklearn.svm import SVC

from sklearn.datasets import load_digits

from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

import math

gnb = GaussianNB()

params = {}

train_and_evaluate(gnb, X_train, X_test, y_train, y_test)



from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier



neigh = KNeighborsClassifier(n_neighbors=5)

train_and_evaluate(neigh, X_train, X_test, y_train, y_test)



k_accuracy = list()

for k in range(1,13):

    neigh = KNeighborsClassifier(n_neighbors=k)

    neigh.fit(X_train, y_train)

    y_pred_knn = neigh.predict(X_test)

    k_accuracy.append(accuracy_score(y_test, y_pred_knn))

    

plt.plot(range(1,13),k_accuracy)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Accuracy scores')

plt.title('The optimal numbers of Neighbour in KNN')

plt.show()



neigh = KNeighborsClassifier(n_neighbors=11)

train_and_evaluate(neigh, X_train, X_test, y_train, y_test)
from sklearn import svm

from sklearn.svm import SVC



clf = SVC(gamma='auto', kernel='rbf')

train_and_evaluate(clf, X_train, X_test, y_train, y_test)
#function for GridSearchCV with import of GridSearchCV library

from sklearn.model_selection import GridSearchCV

def svc_param_selection(X_train, y_train, nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)

    grid_search.fit(X_train, y_train)

    grid_search.best_params_

    return grid_search.best_params_

    

print(svc_param_selection(X_train, y_train, 5))



#C score

#gamma

clf = SVC(C=10, gamma=0.1, kernel='rbf')

train_and_evaluate(clf, X_train, X_test, y_train, y_test)
title = "Learning Curves (Naive Bayes)"

# Cross validation with 10 iterations to get smoother mean test and train

# score curves, each time with 25% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

plot_learning_curve(gnb, title,X, y, ylim=(0.2, 1.0), cv=cv,)

title = "Learning Curves (KNN)"

plt.show()

plot_learning_curve(neigh, title,X, y, ylim=(0.2, 1.0), cv=cv,)

plt.show()

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.1$)"

plt.show()

plot_learning_curve(clf, title,X, y, ylim=(0.2, 1.0), cv=cv,)
# Importing Modules

from sklearn import datasets

import matplotlib.pyplot as plt



# Dataset Slicing

x_axis = df["alcohol"] 

y_axis = df["sulphates"]  



# Plotting

plt.scatter(x_axis, y_axis, c=y)

plt.show()
from sklearn.cluster import KMeans

from scipy.spatial.distance import cdist



X = df[['alcohol','sulphates','citric acid']]



x = df["alcohol"] 

y = df["sulphates"]

z = df["citric acid"]



# create new plot and data

plt.plot()

X = np.array(list(zip(x, y))).reshape(len(x), 2)

colors = ['b', 'g', 'r']

markers = ['o', 'v', 's']



# k means determine k

distortions = []

K = range(1,15)

for k in K:

    kmeanModel = KMeans(n_clusters=k).fit(X)

    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# Plot the elbow method into the graph

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method')

plt.show()
kmeans = KMeans(n_clusters=4)

kmeans = kmeans.fit(X)

labels = kmeans.labels_



centroids = kmeans.cluster_centers_

plt.scatter(

    x, 

    y,

    c=labels,

    cmap='plasma')

plt.xlabel('alcohol', fontsize=18)

plt.ylabel('sulphates', fontsize=16)
from mpl_toolkits.mplot3d import Axes3D

dim = plt.figure(1, figsize=(8, 6))

ax = Axes3D(dim, rect=[0, 0, 1, 1], elev=48, azim=134)



ax.set_xlabel('alcohol')

ax.set_ylabel('sulphates')

ax.set_zlabel('citric acid')



ax.scatter(x, y, z, c = labels)