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
import numpy as np

import pandas as pd

from time import time

from sklearn.metrics import f1_score



# Read student data

iris_data = pd.read_csv("../input/Iris.csv")

n_flowers = len(iris_data.index)

n_features = iris_data.shape[1]



# TODO: Calculate passing students

n_1 = iris_data[iris_data["Species"]=="Iris-setosa"].count()["Id"]



# TODO: Calculate failing students

n_2 = iris_data[iris_data["Species"]=="Iris-versicolor"].count()["Id"]

n_3 = iris_data[iris_data["Species"]=="Iris-virginica"].count()["Id"]



# TODO: Calculate graduation rate

n1_rate = float(n_1)/float(n_flowers)*100

n2_rate = float(n_2)/float(n_flowers)*100

n3_rate = float(n_3)/float(n_flowers)*100



feature_cols = list(iris_data.columns[:-1])

target_col = iris_data.columns[-1] 



# Show the list of columns



# Separate the data into feature data and target data (X_all and y_all, respectively)

X_all = iris_data[feature_cols]

y_all = iris_data[target_col]



def preprocess_features(X):

    ''' Preprocesses the student data and converts non-numeric binary variables into

        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    

    # Initialize new output DataFrame

    output = pd.DataFrame(index = X.index)



    # Investigate each feature column for the data

    for col, col_data in X.iteritems():

        

        # If data type is non-numeric, replace all yes/no values with 1/0

        if col_data.dtype == object:

            col_data = col_data.replace(['Iris-setosa', 'Iris-versicolor','Iris-virginica'], [1,2,3])



        # If data type is categorical, convert to dummy variables

        if col_data.dtype == object:

            # Example: 'school' => 'school_GP' and 'school_MS'

            col_data = pd.get_dummies(col_data, prefix = col)  

        

        # Collect the revised columns

        output = output.join(col_data)

    

    return output



X_all = preprocess_features(X_all)
import matplotlib.pyplot as plt

from sklearn import cross_validation

from sklearn import tree

%matplotlib inline
import matplotlib.pyplot as pl

import numpy as np

import sklearn.learning_curve as curves

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.cross_validation import ShuffleSplit, train_test_split

from sklearn.neural_network import MLPClassifier



def NN_Curve_Layer(X, y):

    """ Calculates the performance of several models with varying sizes of training data.

        The learning and testing scores for each model are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Generate the training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)



    # Create the figure window

    fig = pl.figure(figsize=(10,7))



    # Create three different models based on max_depth

    for k, layer in enumerate([1,2,3,4]):

        

        # Create a Decision tree regressor at max_depth = depth

        clf = MLPClassifier(hidden_layer_sizes=layer)



        # Calculate the training and testing scores

        sizes, train_scores, test_scores = curves.learning_curve(clf, X, y, \

            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        

        # Find the mean and standard deviation for smoothing

        train_std = np.std(train_scores, axis = 1)

        train_mean = np.mean(train_scores, axis = 1)

        test_std = np.std(test_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)



        # Subplot the learning curve 

        

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Cross-Validation Score')

        ax.fill_between(sizes, train_mean - train_std, \

            train_mean + train_std, alpha = 0.15, color = 'r')

        ax.fill_between(sizes, test_mean - test_std, \

            test_mean + test_std, alpha = 0.15, color = 'g')

        

        # Labels

        ax.set_title('hidden layer = %s'%(layer))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])

    

    # Visual aesthetics

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)



    fig.suptitle('Neural Network Classifier Learning Performances', fontsize = 16, y = 1.03)

    fig.tight_layout()

    fig.show()



def NN_Curve_Action(X, y):

    """ Calculates the performance of several models with varying sizes of training data.

        The learning and testing scores for each model are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Generate the training set sizes increasing by 50

    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)



    # Create the figure window

    fig = pl.figure(figsize=(10,7))



    # Create three different models based on max_depth

    for k, act in enumerate(['identity', 'logistic', 'tanh', 'relu']):

        

        # Create a Decision tree regressor at max_depth = depth

        clf = MLPClassifier(activation=act)



        # Calculate the training and testing scores

        sizes, train_scores, test_scores = curves.learning_curve(clf, X, y, \

            cv = cv, train_sizes = train_sizes, scoring = 'accuracy')

        

        # Find the mean and standard deviation for smoothing

        train_std = np.std(train_scores, axis = 1)

        train_mean = np.mean(train_scores, axis = 1)

        test_std = np.std(test_scores, axis = 1)

        test_mean = np.mean(test_scores, axis = 1)



        # Subplot the learning curve 

        

        ax = fig.add_subplot(2, 2, k+1)

        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')

        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Cross-Validation Score')

        ax.fill_between(sizes, train_mean - train_std, \

            train_mean + train_std, alpha = 0.15, color = 'r')

        ax.fill_between(sizes, test_mean - test_std, \

            test_mean + test_std, alpha = 0.15, color = 'g')

        

        # Labels

        ax.set_title('activation = %s'%(act))

        ax.set_xlabel('Number of Training Points')

        ax.set_ylabel('Score')

        ax.set_xlim([0, X.shape[0]*0.8])

        ax.set_ylim([-0.05, 1.05])

    

    # Visual aesthetics

    ax.legend(bbox_to_anchor=(1.05, 2.05), loc='lower left', borderaxespad = 0.)



    fig.suptitle('Neural Network Classifier Learning Performances', fontsize = 16, y = 1.03)

    fig.tight_layout()

    fig.show()

def NN_Validation_layer(X, y):

    """ Calculates the performance of the model as model complexity increases.

        The learning and testing errors rates are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Vary the max_depth parameter from 1 to 10

    layer = np.arange(1,10)



    # Calculate the training and testing scores

    clf = MLPClassifier()

    train_scores, test_scores = curves.validation_curve(clf, X, y, \

        param_name = "hidden_layer_sizes", param_range = layer, cv = cv, scoring = 'accuracy')



    # Find the mean and standard deviation for smoothing

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Plot the validation curve

    pl.figure(figsize=(7, 5))

    pl.title('Neural Network Classifier Validation Performance')

    pl.plot(max_depth, train_mean, 'o-', color = 'r', label = 'Training Score')

    pl.plot(max_depth, test_mean, 'o-', color = 'g', label = 'Validation Score')

    pl.fill_between(max_depth, train_mean - train_std, \

        train_mean + train_std, alpha = 0.15, color = 'r')

    pl.fill_between(max_depth, test_mean - test_std, \

        test_mean + test_std, alpha = 0.15, color = 'g')

    

    # Visual aesthetics

    pl.legend(loc = 'lower right')

    pl.xlabel('Hidden Layer')

    pl.ylabel('Score')

    pl.ylim([-0.05,1.05])

    pl.show()
NN_Curve_Layer(X_all, y_all)
NN_Curve_Action(X_all, y_all)
NN_Validation_layer(X_all, y_all)
def NN_Validation_layer(X, y):

    """ Calculates the performance of the model as model complexity increases.

        The learning and testing errors rates are then plotted. """

    

    # Create 10 cross-validation sets for training and testing

    cv = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)



    # Vary the max_depth parameter from 1 to 10

    layer = np.arange(1,10)



    # Calculate the training and testing scores

    clf = MLPClassifier()

    train_scores, test_scores = curves.validation_curve(clf, X, y, \

        param_name = "hidden_layer_sizes", param_range = layer, cv = cv, scoring = 'accuracy')



    # Find the mean and standard deviation for smoothing

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)



    # Plot the validation curve

    pl.figure(figsize=(7, 5))

    pl.title('Neural Network Classifier Validation Performance')

    pl.plot(layer, train_mean, 'o-', color = 'r', label = 'Training Score')

    pl.plot(layer, test_mean, 'o-', color = 'g', label = 'Validation Score')

    pl.fill_between(layer, train_mean - train_std, \

        train_mean + train_std, alpha = 0.15, color = 'r')

    pl.fill_between(layer, test_mean - test_std, \

        test_mean + test_std, alpha = 0.15, color = 'g')

    

    # Visual aesthetics

    pl.legend(loc = 'lower right')

    pl.xlabel('Hidden Layer')

    pl.ylabel('Score')

    pl.ylim([-0.05,1.05])

    pl.show()
NN_Validation_layer(X_all, y_all)