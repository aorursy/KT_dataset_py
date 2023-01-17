# as proposed in data tab
def read_data(filename):
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip() # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data

train = read_data("../input/kiwhs-comp-1-complete/train.arff")

dataPoints = [[x,y] for [x, y, z] in train]
dataLabels = [z for [x, y, z] in train]
import numpy as np
import matplotlib.pyplot as plt

# visualize data as point plot
colors = []
xPos = []
yPos = []
for i in range(len(dataLabels)):
    xPos.append(dataPoints[i][0])
    yPos.append(dataPoints[i][1])
    if dataLabels[i] == -1:
        colors.append('red')
    else:
        colors.append('blue')
plt.scatter(xPos, yPos, c = colors, marker='o')
plt.xlabel('X')
plt.xlabel('Y')
plt.title('Given point distribution')
plt.show()
import math
import random

#split data into training and test sets
def splitData(data, labels, split):
    splitPoint = math.floor(len(data)*split)
    combine = list(zip(data, labels))
    random.shuffle(combine)
    data[:], labels[:] = zip(*combine)
    xTrain, yTrain = data[:splitPoint], labels[:splitPoint]
    xTest, yTest = data[splitPoint:], labels[splitPoint:]
    return xTrain, yTrain, xTest, yTest

# normalize data with f(x) = (x - my) / sigma
# standard score
from scipy import stats
def normalize(data):
    x, y = zip(*data)
    x = np.asarray(x)
    x = (x-np.mean(x))/np.std(x)
    y = np.asarray(y)
    y = (y-np.mean(y))/np.std(y)
    data = list(zip(x,y))
    return data

# perform on actual data with a split point 0.67
xTrain, yTrain, xTest, yTest = splitData(dataPoints, dataLabels,0.67)
xTrain = normalize(xTrain)
xTest = normalize(xTest)
# build k-nearest-neighbour (kNN) classifier
# define a metric function (euclidean distance for simplicity)
def metricFunction(n1, n2, nDepth):
    dist = 0
    for i in range (nDepth):
        dist+=pow((n1[i]-n2[i]),2)
    return math.sqrt(dist)

# define a function which examines all neighbours
# and returns 'k' nearest neighbours to the current sample
def createNeighbourhood (trainSet, trainLabel, sample, k):
    area = []
    for i in range(len(trainSet)):
        distance = metricFunction(sample,trainSet[i],len(sample))
        area.append((trainSet[i], trainLabel[i], distance))    
    area.sort(key=lambda n: n[2])
    kNN = []
    for i in range(k):
        kNN.append(area[i])
    return kNN
# define a suitable response function (voting)
# simpleVote returns a simple majority vote of all nN 
def simpleVote(kNN, labels):
    voteUp = 0
    voteDown = 0
    for i in range(len(kNN)):
        if kNN[i][1] == labels[0]:
            voteUp += 1
        else:
            voteDown += 1
    if voteUp >= voteDown:
        return 1
    else:
        return -1
    return 0

# define a wrapper function for the kNN classifier
def kNNClassifier(xTrain, yTrain, xTest, k, labels):
    predictions = []
    for i in range(len(xTest)):
        predictions.append(simpleVote(createNeighbourhood(xTrain, yTrain, xTest[i], k), labels))
    return predictions
# define a accuracy metric
# simple measuring of predictions against correct labels
def accuracyRating(predictions, yTest):
    passed = 0
    for i in range(len(yTest)):
        if predictions[i] == yTest[i]:
            passed += 1
    return (passed/float(len(yTest)))
# find a suitable k
k = 32
labels = [1,-1]
predictions = []
accuracies = []
for i in range (1,k):
    predictions = kNNClassifier(xTrain, yTrain, xTest, i, labels)
    accuracies.append(accuracyRating(predictions, yTest))

print(accuracies)
#visualize different k
plt.plot(accuracies)
plt.xlabel("Sampled k")
plt.ylabel("Accuracy")
plt.title("Different sizes of k")
plt.show()
# perform kNN on actual testSet
import csv

def read_csv(filename):
    with open(filename, 'r') as file:
        data = []
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            objects = []
            for i in range (len(row)): 
                objects.append(float(row[i]))
            data.append(objects)
        return data

testSet = read_csv('../input/kiwhs-comp-1-complete/test.csv')
testID = [int(x) for [x, y, z] in testSet]
testPos = [[y,z] for [x, y, z] in testSet]

# use optimal 'k' that we found earlier
k = 14
labels = [1,-1]
predict = []

testPos = normalize(testPos)
predict = kNNClassifier(xTrain, yTrain, testPos, 14, labels)
#create final submission file
submission = list(zip(testID, predict))

def write_csv(filename, file):
    with open(filename, 'w', newline='') as csvfile:
        filewrite = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewrite.writerow(['Id (String)', 'Category (String)'])
        for i in range(len(file)):
            filewrite.writerow([str(file[i][0]),str(file[i][1])])
    
write_csv('submission.csv',submission)
