from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import learning_curve, GridSearchCV

import pandas as pd

import numpy as np

from sklearn import svm, datasets

from sklearn.svm import SVC

import matplotlib.pyplot as plt

import csv

import math

import seaborn as sns

%matplotlib inline

import random



from sklearn.preprocessing import LabelEncoder

from sklearn.cluster import KMeans

import numpy as numpy 

import statistics



#import seaborn as sns
def loadCsv():

	lines = csv.reader(open('../input/dataset2/dd1_h.csv'))

	dataset = list(lines)

	for i in range(len(dataset)):

		dataset[i] = [float(x) for x in dataset[i]]

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

    for x in range(len(testSet)):

        if testSet[x][-1] == predictions[x]:

            correct += 1

    return (correct/float(len(testSet))) * 100.0

def main():

    filename = '../input/dataset2/dd1_h.csv'

    splitRatio = 0.67

    dataset = loadCsv()

    trainingSet, testSet = splitDataset(dataset, splitRatio)

    #dataset = loadCsv(filename)

    trainingSet =loadCsv()

    testSet =loadCsv()

    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))

    summaries = summarizeByClass(trainingSet)

    predictions = getPredictions(summaries, testSet)

    accuracy = getAccuracy(testSet, predictions)

    print('Accuracy: {0}%'.format(accuracy))

main()
train = pd.read_csv('../input/dataset1/dd1.csv')

train.head()
train.count()
train.describe()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Churn',axis=1), 

           train['Churn'], test_size=0.30, 

            random_state=101)

from sklearn.linear_model import LogisticRegression#create an instance and fit the model 

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

#predictions

Predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,Predictions))
from sklearn.decomposition import PCA as sklearnPCA
df = pd.read_csv('../input/dataset1/dd1.csv')

df.columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

df.dropna(how="all", inplace=True) # drops the empty line at file-end

X = df.iloc[:,0:19].values

y = df.iloc[:,19].values





X_std = StandardScaler().fit_transform(X)

sklearn_pca = sklearnPCA(n_components=6) #Reducing to six principal components

Y_sklearn = sklearn_pca.fit_transform(X_std)

print(Y_sklearn)



a= pd.DataFrame(Y_sklearn)

print(a)
dset = pd.read_csv('../input/pcads6/ds6.csv')

print(dset.describe())



y = dset['Churn']

X = dset.drop(['Churn'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=98)
from sklearn.neural_network import MLPClassifier
dset = pd.read_csv("../input/dataset1/dd1.csv")

#dset = pd.read_csv("diabetes.csv")

X=dset.iloc[:, :-1].values

y = dset.iloc[:,19:].values

print(y)

y=y.astype('int')

X=X.astype('int')



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 1/3)

 

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



clf = MLPClassifier(hidden_layer_sizes=(7,9,4,6,8,10,16), max_iter=300, alpha=0.00000000000000000000001,

                     solver='sgd', verbose=3,  random_state=0,tol=0.000000001)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("accuracy",accuracy_score(y_test, y_pred)*100)



cm = confusion_matrix(y_test, y_pred)

print(cm)
Data= pd.read_csv("../input/pcads6/ds6.csv", sep= None, engine= "python")

cols= ["0","1","3","4","5","Churn"]

data_encode= Data.drop(cols, axis= 1)

data_encode= data_encode.apply(LabelEncoder().fit_transform)

data_rest= Data[cols]

Data= pd.concat([data_rest,data_encode], axis= 1)

data_encode= Data.drop(cols, axis= 1)

data_encode= data_encode.apply(LabelEncoder().fit_transform)

data_rest= Data[cols]

Data= pd.concat([data_rest,data_encode], axis= 1)
data_train, data_test= train_test_split(Data, test_size= 0.33, random_state= 4)

X_train= data_train.drop("Churn", axis= 1)

Y_train= data_train["Churn"]

X_test= data_test.drop("Churn", axis=1)

Y_test= data_test["Churn"]
scaler= StandardScaler()

scaler.fit(X_train)

X_train= scaler.transform(X_train)

X_test= scaler.transform(X_test)
#K-means clustering

K_cent= 8 

km= KMeans(n_clusters= K_cent, max_iter= 100)

km.fit(X_train)

cent= km.cluster_centers_
max=0 

for i in range(K_cent):

	for j in range(K_cent):

		d= numpy.linalg.norm(cent[i]-cent[j])

		if(d> max):

			max= d

d= max

 

sigma= d/math.sqrt(2*K_cent)
shape= X_train.shape

row= shape[0]

column= K_cent

G= numpy.empty((row,column), dtype= float)

for i in range(row):

    for j in range(column):

        dist= numpy.linalg.norm(X_train[i]-cent[j])

        G[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))

        
GTG= numpy.dot(G.T,G)

GTG_inv= numpy.linalg.inv(GTG)

fac= numpy.dot(GTG_inv,G.T)

W= numpy.dot(fac,Y_train)
row= X_test.shape[0]

column= K_cent

G_test= numpy.empty((row,column), dtype= float)

for i in range(row):

	for j in range(column):

		dist= numpy.linalg.norm(X_test[i]-cent[j])

		G_test[i][j]= math.exp(-math.pow(dist,2)/math.pow(2*sigma,2))
prediction= numpy.dot(G_test,W)

prediction= 0.5*(numpy.sign(prediction-0.5)+1)

score= accuracy_score(prediction,Y_test)

print ("Accuracy is:")

print (score)