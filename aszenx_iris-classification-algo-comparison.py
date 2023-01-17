import pandas as pd
# the features (cols) in the dataset
col_header=['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)', 'iris class(cm)']

# read file with no headers and col names as mentioned in the above array
# the data is in pandas dataframe type
iris_panda_set = pd.read_csv('../input/irisdata/iris.csv', header=None, names=col_header)
iris_panda_set.head(10)
# change all categorical types to numeric ones
print(iris_panda_set.loc[0, :])
print(iris_panda_set.loc[60, :])
print(iris_panda_set.loc[120, :])

# map iris class types to numeric values
iris_class_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2 }
# change the dataset accordingly
iris_panda_set['iris class(cm)'] = iris_panda_set['iris class(cm)'].map(iris_class_map)
print(iris_panda_set.loc[0, :])
print(iris_panda_set.loc[60, :])
print(iris_panda_set.loc[120, :])
# Stratified ShuffleSplit cross-validator
# Provides train/test indices to split data in train/test sets
# This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.
# The folds are made by preserving the percentage of samples for each class.
from sklearn.model_selection import StratifiedShuffleSplit # import training test split method from sklearn
# next we define the feature cols and predicted col
feature_col_names = ['sepal length(cm)', 'sepal width(cm)', 'petal length(cm)', 'petal width(cm)']
predicted_class_names = ['iris class(cm)']

# split our data into two data frames one containing the features cols and other with the iris category
X = iris_panda_set[feature_col_names].values # predictor feature cols (4)
# predicated class Iris-setosa: 0, Iris-versicolor: 1, Iris-virginica: 2
Y = iris_panda_set[predicted_class_names].values # one-d array

split_test_size = 0.30 # define the train_test split ratio 30%

sss = StratifiedShuffleSplit(n_splits=3, test_size=split_test_size, random_state=20)

for train_index, test_index in sss.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
#  we check to ensure we have the desired 70% train and 30% test split of the data
#  here df.index is the whole data frame
print('{0:0.2f}% in training set'.format((len(X_train) / len(iris_panda_set.index)) * 100))
print('{0:0.2f}% in test set'.format((len(X_test) / len(iris_panda_set.index)) * 100))
# We also verify the predicted values were split the same b/w the train & test data sets
num_setosa = len(iris_panda_set.loc[iris_panda_set['iris class(cm)'] == 0.0])
num_versicolor = len(iris_panda_set.loc[iris_panda_set['iris class(cm)'] == 1.0])
num_virginica = len(iris_panda_set.loc[iris_panda_set['iris class(cm)'] == 2.0])
total = num_setosa + num_versicolor + num_virginica
percentage_setosa = (num_setosa / total) * 100
percentage_versicolor = (num_versicolor / total) * 100
percentage_virginica = (num_virginica / total) * 100
print('Number of setosa plants: {0} ({1:2.2f}%)'.format(num_setosa, percentage_setosa))
print('Number of versicolor plants: {0} ({1:2.2f}%)'.format(num_versicolor, percentage_versicolor))
print('Number of virginica plants: {0} ({1:2.2f}%)'.format(num_virginica, percentage_virginica))

num_setosa_in_train = len(Y_train[Y_train[:] == 0.0])
num_versicolor_in_train = len(Y_train[Y_train[:] == 1.0])
num_virginica_in_train = len(Y_train[Y_train[:] == 2.0])
total_in_train = num_setosa_in_train + num_versicolor_in_train + num_virginica_in_train
percentage_setosa_in_train = (num_setosa_in_train / total_in_train) * 100
percentage_versicolor_in_train = (num_versicolor_in_train / total_in_train) * 100
percentage_virginica_in_train = (num_virginica_in_train / total_in_train) * 100

# printing the result
print('Number of setosa plants in train set: {0} ({1:2.2f}%)'.format(num_setosa_in_train, percentage_setosa_in_train))
print('Number of versicolor plants in train set : {0} ({1:2.2f}%)'.format(num_versicolor_in_train, percentage_versicolor_in_train))
print('Number of virginica plants in train set: {0} ({1:2.2f}%)'.format(num_virginica_in_train, percentage_virginica_in_train))

num_setosa_in_test = len(Y_test[Y_test[:] == 0.0])
num_versicolor_in_test = len(Y_test[Y_test[:] == 1.0])
num_virginica_in_test = len(Y_test[Y_test[:] == 2.0])
total_in_test = num_setosa_in_test + num_versicolor_in_test + num_virginica_in_test
percentage_setosa_in_test = (num_setosa_in_test / total_in_test) * 100
percentage_versicolor_in_test = (num_versicolor_in_test / total_in_test) * 100
percentage_virginica_in_test = (num_virginica_in_test / total_in_test) * 100

# printing the result
print('Number of setosa plants in test set: {0} ({1:2.2f}%)'.format(num_setosa_in_test, percentage_setosa_in_test))
print('Number of versicolor plants in test set : {0} ({1:2.2f}%)'.format(num_versicolor_in_test, percentage_versicolor_in_test))
print('Number of virginica plants in test set: {0} ({1:2.2f}%)'.format(num_virginica_in_test, percentage_virginica_in_test))
# import Naive Bayes algorithm from the library
# In case of naive_bayes there are multiple implementations 
# we are using the gaussian algo that assumes that the feature data is distributed in a gaussian 
from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with data
nb_model = GaussianNB() # our model object

# call the fit method to create a model trained with the training data 
# numpy.ravel returns a contiguous flattened array
nb_model.fit(X_train, Y_train.ravel())
# to see the accuracy we load the scikit metrics library
# metrics has methods that let us get the statistics on the models predictive performance
from sklearn import metrics
# Now lets predict against the testing data
# X_test is the data we kept aside for testing
nb_predict_test = nb_model.predict(X_test)
# Y_test is the actual output and nb_predict_test is the predicted one 
test_accuracy = metrics.accuracy_score(Y_test, nb_predict_test)
print('Accuracy(%) of NB model on test data: {0: .4f}'.format(test_accuracy * 100))
# the classification report generates statistics based on the values shown in the confusion matrix.
print('\nClassification report of NB model:\n')
print(metrics.classification_report(Y_test, nb_predict_test, labels = [0, 1, 2]))
print('Here:\n 0 -> Iris-setosa \n 1 -> Iris-versicolor \n 2 -> Iris-virginica ')
# import random forest from scikit
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state = 54) # Create random forest object
rf_model.fit(X_train, Y_train.ravel())
rf_predict_test = rf_model.predict(X_test)
# training metrics
print('Accuracy(%) of RF model on test data: {0:.4f}'.format((metrics.accuracy_score(Y_test, rf_predict_test)) * 100))
print('\nClassification report of RF model:\n')
print(metrics.classification_report(Y_test, rf_predict_test, labels = [0, 1, 2]))
print('Here:\n 0 -> Iris-setosa \n 1 -> Iris-versicolor \n 2 -> Iris-virginica ')
# scikit learn has an ensemble algorithm that combines logistic regression with cross validation called LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV
lr_cv_model = LogisticRegressionCV(Cs=3, refit=True, cv=10,)
lr_cv_model.fit(X_train, Y_train.ravel())
lr_cv_predict_test = lr_cv_model.predict(X_test)
# training metrics
print('Accuracy(%) of LRCV model on test data: {0:.4f}'.format((metrics.accuracy_score(Y_test, lr_cv_predict_test)) * 100))
print('\nClassification report of LRCV model:\n ')
print(metrics.classification_report(Y_test, lr_cv_predict_test, labels = [0, 1, 2]))
print('Here:\n 0 -> Iris-setosa \n 1 -> Iris-versicolor \n 2 -> Iris-virginica ')
# import the algorithm
# Multi-layer Perceptron (MLP) is a supervised learning algorithm
# MLPClassifier implements a multi-layer perceptron (MLP) algorithm 
    # that trains using Backpropagation.
from sklearn.neural_network import MLPClassifier
# initialize the MLP classifier model with parameters
nn_model = MLPClassifier(hidden_layer_sizes=((3, 3, 3)), max_iter=20000)
# train the model with the training set inputs and outputs 
nn_model.fit(X_train, Y_train.ravel())
nn_predict_test = nn_model.predict(X_test)
# training metrics
print('Accuracy(%) of MLP model on test data: {0:.4f}'.format((metrics.accuracy_score(Y_test, nn_predict_test)) * 100))
print('\nClassification report of MLP model:\n')
print(metrics.classification_report(Y_test, nn_predict_test, labels = [0, 1, 2]))
print('Here:\n 0 -> Iris-setosa \n 1 -> Iris-versicolor \n 2 -> Iris-virginica ')
from sklearn import svm
# Create a classifier: a support vector classifier
svm_model = svm.SVC(gamma=0.001)
svm_model.fit(X_train, Y_train.ravel())
svm_predict_test = svm_model.predict(X_test)
# training metrics
print('Accuracy(%) of SVM model on test data: {0:.4f}'.format((metrics.accuracy_score(Y_test, svm_predict_test)) * 100))
print('\nClassification report of SVM model:\n')
print(metrics.classification_report(Y_test, svm_predict_test, labels = [0, 1, 2]))
print('Here:\n 0 -> Iris-setosa \n 1 -> Iris-versicolor \n 2 -> Iris-virginica ')