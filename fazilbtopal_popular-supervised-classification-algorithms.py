from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import *  



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

%matplotlib inline 



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/heart.csv')

df.head()
df.info()
X = df.drop(['target', ], axis=1)

X.head()
y = df['target']

y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
def plot_confusion_matrix(y, pred, labels, classes, normalize=False, cmap=plt.cm.Blues):

    """

    Plots the confusion matrix.

    Args:

        y: Data Labels

        pred: Predicted outputs

        labels: A list of label values to calculate confusion matrix

        classes: A list of containing unique class names for plotting

        normalize:Wheter to plot data with int or percentage values. Default is int.

        cmap: Color map pf the plot

    

    """

    cm = confusion_matrix(y, pred, labels=labels)

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title('Confusion Matrix')

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



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def best_model(model, train, test, grid_params):

    """

    Takes a model and grid params as an input and finds the best model.

    

    Args:

        model: A model class

        train: A dict containing train features as X and labels as y

        test: A dict containing test features as X and labels as y

        grid_params: GridSearchCV parameters

        

    Returns:

        best_estimator, table and best_params

    """

    

    grid = GridSearchCV(model, grid_params, cv=4, scoring='f1_weighted', 

                        n_jobs=-1, return_train_score=True).fit(train['X'], train['y'])

    estimator = grid.best_estimator_

    table = pd.DataFrame(grid.cv_results_).loc[:, 

                ['params', 'mean_test_score', 'std_test_score','mean_train_score', 

                 'std_train_score']].sort_values(by='mean_test_score', ascending=False)

    

    params = grid.best_params_

    preds = estimator.predict(test['X'])

    plot_confusion_matrix(test['y'], preds, labels=[1, 0], classes=['target=1','target=0'])

    print(classification_report(test['y'], preds))

    

    return estimator, table, params

    
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()
est, table, params = best_model(lr, train={'X': X_train, 'y': y_train},

                                test={'X': X_test, 'y':y_test},

                                grid_params=[{'solver':['liblinear', 'sag', 'newton-cg', 'lbfgs'],

                                              'C': [0.01, 0.05, 0.1, 0.5, 1, 5]}])
est
params
table
est.predict_proba(X_test)[0:5]
jaccard_similarity_score(y_test, est.predict(X_test))
log_loss(y_test, est.predict_proba(X_test))
from sklearn.svm import SVC



sv = SVC(gamma='scale')
est, table, params = best_model(sv, train={'X': X_train, 'y': y_train},

                                test={'X': X_test, 'y':y_test},

                                grid_params=[{'kernel':['linear', 'rbf'],

                                              'C': [1, 3, 5, 7, 10, 20]}])
est
table
params
jaccard_similarity_score(y_test, est.predict(X_test))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()
est, table, params = best_model(knn, train={'X': X_train, 'y': y_train},

                                test={'X': X_test, 'y':y_test},

                                grid_params=[{'n_neighbors':list(range(5,30)),

                                              'algorithm': ['ball_tree', 'kd_tree', 'brute'],

                                              'leaf_size': [10, 20, 30, 40, 50]}])
est
table.head()
params
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()
est, table, params = best_model(dt, train={'X': X_train, 'y': y_train},

                                test={'X': X_test, 'y':y_test},

                                grid_params=[{'max_depth':list(range(4,15)),

                                              'criterion': ['gini', 'entropy']}])
est
table
params
import matplotlib.image as mpimg

from sklearn import tree



filename = "tree.png"

feature_names = X.columns.tolist()

target_names = ['0', '1']

tree.export_graphviz(est, feature_names=feature_names, out_file='tree.dot', 

                           class_names=target_names, filled=True, 

                           special_characters=True) 
print(os.listdir('../working/'))
! dot -Tpng tree.dot -o tree.png
img = mpimg.imread('../working/tree.png')

plt.figure(figsize=(100, 200))

plt.imshow(img, interpolation='nearest')
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()
est, table, params = best_model(nb, train={'X': X_train, 'y': y_train},

                                test={'X': X_test, 'y':y_test},

                                grid_params=[{'var_smoothing':[1e-2, 1e-3, 1e-4, 1e-5,

                                                               1e-6, 1e-7, 1e-8, 1e-9]}])
est
table
params