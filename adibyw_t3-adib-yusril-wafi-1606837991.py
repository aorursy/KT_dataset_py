%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np
data = pd.read_csv("../input/dataset/diabetes.csv")

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

target = ['Outcome']

data.head()
for column in data:

    sorted_col = data.sort_values(column)

    x_max = sorted_col[column].iloc[-1]

    x_min = sorted_col[column].iloc[0]

    range_x = x_max - x_min

    

    for i in range(data.shape[0]):

        data.loc[i, column] = (data[column][i] - x_min) / range_x

    

data.head()

def manhattan (a, b):

    distance = 0

    

    for column in features:

        d = a.at[column] - b.at[column]

        distance += abs(d)

    

    return distance



manh_mtx = np.zeros((768,768))

x = data[features]



for i in range(768):

    for j in range(i,768):

        if manh_mtx[i][j] != 0:

            continue

        

        manh_mtx[i][j] = manhattan(x.loc[i],x.loc[j])

        manh_mtx[j][i] = manh_mtx[i][j]



manh_mtx
def euclidean (a, b):

    distance = 0

    

    for column in features:

        d = a.at[column] - b.at[column]

        distance += np.power(d,2)

    

    return np.power(distance, 0.5)



eucl_mtx = np.zeros((768,768))

x = data[features]



for i in range(768):

    for j in range(i,768):

        if eucl_mtx[i][j] != 0:

            continue

        

        eucl_mtx[i][j] = euclidean(x.loc[i],x.loc[j])

        eucl_mtx[j][i] = eucl_mtx[i][j]



eucl_mtx
def cosine (a,b):

    distance = 0

    len_a = 0

    len_b = 0

    

    for column in features:

        distance += a.at[column] * b.at[column]

        len_a += np.power(a.at[column],2)

        len_b += np.power(b.at[column],2)

    

    pembagi = np.power(len_a, 0.5) * np.power(len_b, 0.5)

    return distance / pembagi

    

cos_mtx = np.zeros((768,768))

x = data[features]



for i in range(768):

    for j in range(i,768):

        if cos_mtx[i][j] != 0:

            continue

        

        cos_mtx[i][j] = cosine(x.loc[i],x.loc[j])

        cos_mtx[j][i] = cos_mtx[i][j]



cos_mtx
sample = data.sample(10)

sample
import operator

    

def knn (data, k):

    index = data.index

    

    manh_sample = []

    eucl_sample = []

    cos_sample = []

    

    # Get distance from previous calculation

    for i in index:

        manh_sample.append([])

        eucl_sample.append([])

        cos_sample.append([])

        

        for j in index:

            manh_sample[-1].append((i, j, manh_mtx[i][j]))

            eucl_sample[-1].append((i, j, eucl_mtx[i][j]))

            cos_sample[-1].append((i, j, cos_mtx[i][j]))

    

    # Sort the distance

    for i in range(len(manh_sample)):

        manh_sample[i].sort(key=operator.itemgetter(2))

        eucl_sample[i].sort(key=operator.itemgetter(2))

        cos_sample[i].sort(key=operator.itemgetter(2))

    

    # Get K neighbours

    manh_nghb = []

    eucl_nghb = []

    cos_nghb = []

    

    for i in range(10):

        index = manh_sample[i][0][0]

        manh_nghb.append((index,[]))

        eucl_nghb.append((index,[]))

        cos_nghb.append((index,[]))

        

        for j in range(1,k+1):

            manh_nghb[-1][1].append(manh_sample[i][j][1])

            eucl_nghb[-1][1].append(eucl_sample[i][j][1])

            cos_nghb[-1][1].append(cos_sample[i][j][1])

    

    # Get the predicted outcome

    manh_predict = {}

    eucl_predict = {}

    cos_predict = {}

    

    for i in range(10):

        index = manh_nghb[i][0]

        manh_predict[index] = {}

        eucl_predict[index] = {}

        cos_predict[index] = {}

        

        for j in range(k):

            # Predict using Manhattan Distance

            neighbour = manh_nghb[i][1][j]

            predict = str(sample.loc[neighbour]['Outcome'])

            

            if predict in  manh_predict[index]:

                manh_predict[index][predict] += 1

            else:

                manh_predict[index][predict] = 1

            

            # Predict using Euclidean Distance

            neighbour = eucl_nghb[i][1][j]

            predict = str(sample.loc[neighbour]['Outcome'])

            

            if predict in  eucl_predict[index]:

                eucl_predict[index][predict] += 1

            else:

                eucl_predict[index][predict] = 1

            

            # Predict using Cosine Distance

            neighbour = cos_nghb[i][1][j]

            predict = str(sample.loc[neighbour]['Outcome'])

            

            if predict in  cos_predict[index]:

                cos_predict[index][predict] += 1

            else:

                cos_predict[index][predict] = 1

    

    for i in sample.index:

        sort_out = sorted(manh_predict[i].items(), key=operator.itemgetter(1), reverse=True)

        manh_predict[i] = sort_out[0][0]

        

        sort_out = sorted(eucl_predict[i].items(), key=operator.itemgetter(1), reverse=True)

        eucl_predict[i] = sort_out[0][0]

        

        sort_out = sorted(cos_predict[i].items(), key=operator.itemgetter(1), reverse=True)

        cos_predict[i] = sort_out[0][0]

            

    

    for i in sample.index:

        actual = f"Actual outcome for index {i} is {sample.loc[i]['Outcome']}"

        predicted = f"Predicted outcome for index {i}: Manhattan = {manh_predict[i]}, Euclidean = {eucl_predict[i]}, Cosine =  {cos_predict[i]}"

        print(actual)

        print(predicted)

        

    manh_accuracy = 0

    eucl_accuracy = 0

    cos_accuracy = 0

    

    for i in sample.index:

        actual = str(sample.loc[i]["Outcome"])

        

        if actual ==  manh_predict[i]:

            manh_accuracy += 1

        if actual ==  eucl_predict[i]:

            eucl_accuracy += 1

        if actual ==  cos_predict[i]:

            cos_accuracy += 1

    

    manh_accuracy = (manh_accuracy / 10) * 100

    eucl_accuracy = (eucl_accuracy / 10) * 100

    cos_accuracy = (cos_accuracy / 10) * 100

    

    print(f"Accuracy using Manhattan Distance = {manh_accuracy}")

    print(f"Accuracy using Euclidean Distance = {eucl_accuracy}")

    print(f"Accuracy using Cosine Distance = {cos_accuracy}")

    



print("K = 3")

knn(sample, 3)

print("="*50)

print("K = 5")

knn(sample, 5)

print("="*50)

print("K = 7")

knn(sample, 7)
def mean (data):

    return sum(data)/len(data)



def stdev(data):

    avg = mean(data)

    variance = sum([pow(x-avg,2) for x in data])/float(len(data)-1)

    return np.sqrt(variance)



def separateByClass(dataset):

    separated = {}

    for i in range(len(dataset)):

        vector = dataset[i]

        if (vector[-1] not in separated):

            separated[vector[-1]] = []

        separated[vector[-1]].append(vector)

    return separated



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

    exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(stdev,2))))

    return (1 / (np.sqrt(2*np.pi) * stdev)) * exponent



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
training = data.loc[:(768 * 80 // 100)]

test = data.loc[(768 * 80 // 100 + 1):]



# Prepare the model

summaries = summarizeByClass(training.values)



# Test the model

predictions = getPredictions(summaries, test.values)



index = 0

for i in test.index:

    print(f"Actual Outcome for index {i} = {test.loc[i]['Outcome']}, Predicted Outcome = {predictions[index]}")

    index += 1
def getAccuracy(testSet, predictions):

    correct = 0

    for x in range(len(testSet)):

        if testSet[x][-1] == predictions[x]:

            correct += 1

    return (correct/float(len(testSet))) * 100.0



accuracy = getAccuracy(test.values, predictions)

print(f"Accuracy = {accuracy}")