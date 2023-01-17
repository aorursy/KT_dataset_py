#Importing

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier



from sklearn.metrics import accuracy_score



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Load data

np.random.seed(1)



df2 = pd.read_csv("../input/sensor_readings_2.csv", header = None)

df4 = pd.read_csv("../input/sensor_readings_4.csv", header = None)

df24 = pd.read_csv("../input/sensor_readings_24.csv", header = None)





classes = ("Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn")

n_classes = len(classes)



for i, item in enumerate(classes):

    df2 = df2.replace(to_replace = item, value = i)

    df4 = df4.replace(to_replace = item, value = i)

    df24 = df24.replace(to_replace = item, value = i)



df = df24

models = ("GNB", "linearSVM", "rbfSVM", "NeuralNet")
# Select random part as train and test sets

msk = np.random.rand(len(df)) < 0.8

dfTrain = df[msk]

dfTest = df[~msk]



# Convert to numpy arrays for scikit learn

dataTrain = np.array(dfTrain)

dataTest = np.array(dfTest)



# Divide in X and Y

XTrain = dataTrain[:,0:dataTrain.shape[1]-1]

YTrain = dataTrain[:,dataTrain.shape[1]-1]



XTest = dataTest[:,0:dataTrain.shape[1]-1]

YTest = dataTest[:,dataTrain.shape[1]-1]
# Run models

for model in models:



    if model == "GNB":

        # Gaussian Navie Bayes

        clf = GaussianNB()

    

    elif model == "linearSVM":

        clf = SVC(kernel = 'linear')

        

    elif model == "rbfSVM":

        clf = SVC(kernel = 'rbf')

    

    elif model == "NeuralNet":

        clf = MLPClassifier()



    clf.fit(XTrain, YTrain)        

    YPred = clf.predict(XTest)

    YPredTrain = clf.predict(XTrain)

    trainAccuracy = accuracy_score(YTest, YPred, normalize = True)

    testAccuracy = accuracy_score(YTrain, YPredTrain, normalize = True)

    print("Model: " + str(model))

    print("Train accuracy: "  + str(trainAccuracy))

    print("Test accuracy: "  + str(testAccuracy))

    print("")



#plt.clf()

#handles = plt.plot(np.column_stack((YTest, YPred)))

#plt.legend(handles, ['True direction', 'Predicted'])

#plt.title(model)