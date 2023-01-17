# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.metrics import accuracy_score
import csv
import random
import math

dataset = pd.read_csv('../input/iris.csv')
feature_col = ['sepal_length', 'sepal_width', 'petal_length','petal_width']
X = dataset[feature_col].values
y = dataset['species'].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
class KNNClassifier():
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    def fitting(self, X, y):
        
        n_samples = X.shape[0]
        # number of neighbors can't be more then number of samples
        if self.n_neighbors > n_samples:
            raise ValueError("number of neighbors will not be more then samples.")
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y
        
    def prediction(self, X_test):
        
        # number of predictions to make and number of features inside single sample
        n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
        for i in range(n_predictions):
            # calculation of single prediction
            predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

        return(predictions)

def single_prediction(X, y, x_train, k):
    
    # number of samples inside training set
    n_samples = X.shape[0]
    
    # create array for distances and targets
    distances = np.empty(n_samples, dtype=np.float64)
    
    # distance calculation
    for i in range(n_samples):
        distances[i] = (x_train - X[i]).dot(x_train - X[i])
    
    
    # combining arrays as columns
    distances = sp.c_[distances, y]
    # sorting array by value of first column
    sorted_distances = distances[distances[:,0].argsort()]
    
    # celecting labels associeted with k smallest distances
    targets = sorted_distances[0:k,1]
    
    unique, counts = np.unique(targets, return_counts=True)
    return(unique[np.argmax(counts)])
classifier = KNNClassifier(n_neighbors=5)

# Fitting the model on training data
classifier.fitting(X_train, y_train)

# Predicting the Test set results

my_y_pred = classifier.prediction(X_test)
accuracy = accuracy_score(y_test, my_y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
x_train = [5.6, 3, 4.1, 1.3]
k = 3
res=single_prediction(X, y, x_train, k)
print('Sample data benlongs to class ' +str(res))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=5)
# Fitting the model
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
def loadCsv(f_name):
    df=pd.read_csv(f_name)
    species=df["species"]
    species_encoded,speices_cat=species.factorize()
    df["species"]=species_encoded
    dataset=df.as_matrix(columns=None)
    dataset=dataset.tolist()
    return dataset
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions
def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
f_name = '../input/iris.csv'
splitRatio = 0.67
dataset = loadCsv(f_name)
trainingSet, testSet = splitDataset(dataset, splitRatio)
summaries = summarizeByClass(trainingSet)
# test model
predictions = getPredictions(summaries, testSet)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: '+str(accuracy))
from subprocess import check_output
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings 
warnings.filterwarnings('ignore')
from math import ceil
#Plots
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix #Confusion matrix
from sklearn.metrics import accuracy_score  
from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import parallel_coordinates
#Advanced optimization
from scipy import optimize as op

#Load Data
iris = pd.read_csv('../input/iris.csv')
iris.head()
#iris['species']=iris['species'].astype('category')
#iris.info()
#Visualizations

#Plot with respect to sepal length
sepalPlt = sb.FacetGrid(iris, hue="species", size=6).map(plt.scatter, "sepal_length", "sepal_width")
plt.legend(loc='upper left');
plt.show()
#plot with respect to petal length
petalPlt = sb.FacetGrid(iris, hue="species", size=6).map(plt.scatter, "petal_length", "petal_width")
plt.legend(loc='upper left');
plt.show()
#Sepal and Petal lengths
parallel_coordinates(iris, "species");
plt.show()
#Data setup

species = ['setosa', 'versicolor', 'virginica']
#Number of examples
m = iris.shape[0]
#Features
n = 4
#Number of classes
k = 3

X = np.ones((m,n + 1))
y = np.array((m,1))
X[:,1] = iris['sepal_length'].values
X[:,2] = iris['sepal_width'].values
X[:,3] = iris['petal_length'].values
X[:,4] = iris['petal_width'].values

#Labels
y = iris['species'].values

#Mean normalization
for j in range(n):
    X[:, j] = (X[:, j] - X[:,j].mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

#Logistic Regression

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

#Regularized cost function
def regCostFunction(theta, X, y, _lambda = 0.0001):
    m = len(y)
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0 
    reg = (_lambda/(2*m)) * np.sum(tmp**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

#Regularized gradient function
def regGradient(theta, X, y, _lambda = 0.0001):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = _lambda*tmp /m

    return ((1 / m) * X.T.dot(h - y)) + reg

#Optimal theta 
def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)
    
    return result.x


#Training

all_theta = np.zeros((k, n + 1))

#One vs all
i = 0
for flower in species:
    #set the labels in 0 and 1
    tmp_y = np.array(y_train == flower, dtype = int)
    optTheta = logisticRegression(X_train, tmp_y, np.zeros((n + 1,1)))
    all_theta[i] = optTheta
    i += 1
#Predictions
P = sigmoid(X_test.dot(all_theta.T)) #probability for each flower
p = [species[np.argmax(P[i, :])] for i in range(X_test.shape[0])]

print("Test Accuracy ", accuracy_score(y_test, p) * 100 , '%')
#Confusion Matrix
cfm = confusion_matrix(y_test, p, labels = species)

sb.heatmap(cfm, annot = True, xticklabels = species, yticklabels = species);
plt.show();

import numpy as np # linear algebra
import pandas as pd # data processing

from subprocess import check_output
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))

#print(iris)

X = iris["data"][:,3:]  # petal width
y = (iris["target"]==2).astype(np.int)

log_reg = LogisticRegression(penalty="l2")
log_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)

plt.plot(X,y,"b.")
plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris-Virginca")
plt.xlabel("Petal width", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.show()

log_reg.predict([[1.7],[1.5]])
iris = datasets.load_iris()

X = iris["data"][:,(2,3)]  # petal length, petal width
y = iris["target"]

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)
softmax_reg.fit(X,y)

X_new = np.linspace(0,3,1000).reshape(-1,2)

plt.plot(X[:, 0][y==1], X[:, 1][y==1], "y.", label="Iris-Versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.", label="Iris-Setosa")

plt.legend(loc="upper left", fontsize=14)

plt.show()
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 11)

y_train = np.reshape(y_train,(y_train.shape[0],1))
y_test = np.reshape(y_test,(y_test.shape[0],1))
X_test.shape
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
model = LogisticRegression(C=1000000.)
model = model.fit(X_train,y_train)
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
scoring = 'accuracy'
#print('coef', model.coef_)
#print('intercept', model.intercept_)
results = model_selection.cross_val_score(model, X_test,y_test , cv=kfold, scoring=scoring)
print("Accuracy:" ,results.mean()*100,'%')
def sigmoid(z):
    
    return 1.0 / (1.0 + np.exp(-1.0 * z))

    
#Regularized cost function

def regCostFunction(theta, X, y, _lambda = 0.01):
    m = X.shape[0]
    h = sigmoid(np.dot(X_train,theta))
    tmp = np.copy(theta)
    tmp[0] = 0 
    reg = (_lambda/(2*m)) * np.sum(tmp**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

#Regularized gradient function

def regGradient( X, y,theta, _lambda ):
    m=X.shape[0]
    n=X.shape[1]
    h = sigmoid(np.dot(X_train,theta))
    tmp = np.copy(theta)
    tmp[0] = 0
    reg = _lambda*tmp /m

    return ( (1/m)* X.T.dot(h - y)) 
def  Newton_GradientDescent (X,y,theta,hessian,itera,lamb):
    cost=[]
    for i in range(itera):
        l=sigmoid(np.dot(X_train,theta))
        error = l-y
        grad = regGradient(X,y,theta,lamb)
        theta = theta - np.dot(hessian,grad)
    return theta
theta = np.zeros((X_train.shape[1],1))
from sklearn.metrics import accuracy_score
score = sigmoid(np.dot(X_train,theta))
sigma = np.diag((score*(1-score))[:,0])
#print(sigma)
temp = np.dot(sigma,X_train) 
hessian =np.linalg.inv((1/X_train.shape[0])*np.dot(X_train.T,temp))

opt_theta = Newton_GradientDescent(X_train,y_train,theta,hessian,10,0.4)
P = np.where(sigmoid(np.dot(X_test,opt_theta))>=0.5,1,0)
print("Test Accuracy",accuracy_score(y_test,P)*100,'%')

