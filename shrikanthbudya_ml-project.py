# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB

import math
data = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")
model = GaussianNB()
data.dropna(inplace=True)

properties = data[["MinTemp","MaxTemp","Rainfall","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]]

label = data[["RainTomorrow"]]
properties["RainToday"].replace(to_replace={"No":0,"Yes":1},inplace=True)

label["RainTomorrow"].replace(to_replace={"No":0,"Yes":1},inplace=True)
model.fit(properties,label)
s=data.sample(100)

psample = s[["MinTemp","MaxTemp","Rainfall","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm","RainToday"]]

lsample = s[["RainTomorrow"]]

#properties.replace([-np.inf,np.inf],np.nan)

psample["RainToday"].replace(to_replace={"No":0,"Yes":1},inplace=True)

lsample["RainTomorrow"].replace(to_replace={"No":0,"Yes":1},inplace=True)

pred=model.predict(psample)

count=0

for i in range(len(psample)):

    if pred[i] == lsample["RainTomorrow"].iloc[i]:

        count+=1

print(count)


class GaussianNaiveBayes():

    def __init__(self):

        self.summaries = {}



    def separate_by_class(self, X, y):

        separated = {}

        for i in range(len(X)):

            if y[i] in separated:

                separated[y[i]].append(X[i])

            else:

                separated[y[i]] = [X[i]]

        return separated



    def mean(self, numbers):

        return sum(numbers)/float(len(numbers))

    

    def stdev(self, numbers):

        avg = self.mean(numbers)

        variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)

        return math.sqrt(variance)



    def summarize(self, dataset):

        summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]

        return summaries



    def summarize_by_class(self, separated):

        for classValue, instances in separated.items():

            self.summaries[classValue] = self.summarize(instances)

        return self.summaries



    def fit(self, X, y):

        #X = X.tolist()

        #y = y.tolist()

        separated = self.separate_by_class(X, y)

        self.summaries = self.summarize_by_class(separated)

        #print(self.summaries)



    def calculate_probability(self, x, mean, stdev):

        if stdev == 0:

            return 1

        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))

        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent



    def calculate_probabilites(self, summaries, input_vec):

        probabilities = {}

        for classValue, classSummaries in self.summaries.items():

            probabilities[classValue] = 1

            for i in range(len(classSummaries)):

                mean, stdev = classSummaries[i]

                x = input_vec[i]

                probabilities[classValue] *= self.calculate_probability(x, mean, stdev)

        return probabilities



    def predict(self, input_data):

        final_probabilities = []

        for data in input_data:

            probabilities = self.calculate_probabilites(self.summaries, data)

            final_probabilities.append(probabilities)

            #print(probabilities)

        return final_probabilities



    def predict_classes(self, input_data):

        predictions = []

        f_probs = self.predict(input_data)

        for prob in f_probs:

            pred = max(prob, key = prob.get)

            predictions.append(pred)

        return predictions
cnb = GaussianNaiveBayes()

parr = properties.values

larr = label.values

l1darr = []

for i in larr:

    l1darr.append(i[0])

#print(l1darr)

cnb.fit(parr,l1darr)
psamplearr = psample.values

lsamplearr = lsample.values

cpred = cnb.predict(psamplearr)



ccount = 0

for i in range(len(cpred)):

    #print(cpred[i], " : ", lsamplearr[i][0])

    currentpred = -1

    if cpred[i][0] > cpred[i][1]:

        currentpred = 0

    else:

        currentpred = 1

    if currentpred == lsamplearr[i][0]:

        ccount += 1

print("library accuracy : ",count)

print("custom accuracy  : ",ccount)