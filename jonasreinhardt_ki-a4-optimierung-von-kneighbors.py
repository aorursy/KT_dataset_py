import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
print(os.listdir("../input/")) 
# read the train-data
def read_arff(filename):
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

train_data = read_arff("../input/train.arff") #read train data
train_data = np.array(train_data) # put it in an array
print("datapoints of train data:", len(train_data)) # print the amount of data points
from sklearn import neighbors

def true_accuracy(classes):
    right = 0
    for i,c in enumerate(classes):
        if i < 200:
            if c == -1: right += 1
        else:
            if c == 1: right += 1
    return right/len(classes)

test_data = pd.read_csv("../input/test.csv", index_col=0, header=0, names=['Id (String)', 'X', 'Y']) # get the test-data for evaluation
test_data = np.array(test_data[['X','Y']].values) # put data in numpy array
classifier = neighbors.KNeighborsClassifier() # initialize classifier
# split data over the input arguments and the corresponding class
train_input = train_data[:, 0:2]
train_classes = train_data[:, 2]

print(train_input[0]) # first entry of the input arguments
print(train_classes[0]) # first entry of the corresponding classes
classifier.fit(train_input,train_classes) # train the classifier
predicted_classes = classifier.predict(test_data) # let it run over the test data
# measure and print the accuracy 
accuracy = true_accuracy(predicted_classes)
print("accuracy:", accuracy)
weights = ["uniform", "distance"]
algorithms = ["auto", "ball_tree", "kd_tree", "brute"]

for weight in weights:
    for algo in algorithms:
        classifier = neighbors.KNeighborsClassifier(5, weight, algo) # 5 is the default value for n_neighbors
        classifier.fit(train_input,train_classes) # train the classifier
        predicted_classes = classifier.predict(test_data) # let it run over the test data
        # measure and print the accuracy 
        accuracy = true_accuracy(predicted_classes)
        print("accuracy with weight =", weight, " and algorithm =", algo, ":", accuracy)
        
import matplotlib.pyplot as plt
def plot_datapoints(data):
    x1,y1 = zip(*[(x,y) for x,y,c in data if c == 1])
    x2,y2 = zip(*[(x,y) for x,y,c in data if c == -1])
    plt.plot(x1, y1, '.')
    plt.plot(x2, y2, '.')
    plt.axis('equal')
    plt.axis([-4, 4, -4, 4])

plot_datapoints(train_data)
for k in range(2, 10):
        classifier = neighbors.KNeighborsClassifier(2*k) # puts all even numbers from 4 to 20
        classifier.fit(train_input,train_classes) # train the classifier
        predicted_classes = classifier.predict(test_data) # let it run over the test data
        # measure and print the accuracy 
        accuracy = true_accuracy(predicted_classes)
        print("accuracy with n_neighbors =", 2*k, ":", accuracy)
results = {}
for k in range(1, 399):
    classifier = neighbors.KNeighborsClassifier(k)
    classifier.fit(train_input, train_classes)  # train the classifier
    predicted_classes = classifier.predict(test_data)  # let it run over the test data
    # measure and save the accuracy 
    accuracy = true_accuracy(predicted_classes)
    results.update({k : accuracy})

def plot_barchart(data, from_x, to_x, from_y, to_y):
    plt.bar(range(len(data)), data.values(), align='center')
    plt.axis([from_x, to_x, from_y, to_y])
    fig = plt.gcf()
    fig.set_size_inches(20, 5)

plot_barchart(results, 0, 400, 0.92, 0.98)
plot_barchart(results, 375, 400, 0.955, 0.98)
results = {}

for k in range(1, 399):
    classifier = neighbors.KNeighborsClassifier(k)
    classifier.fit(train_input, train_classes)  # train the classifier
    predicted_classes = classifier.predict(train_data[:,0:2])  # let it run over the train data
    # measure and save the accuracy 
    accuracy = true_accuracy(predicted_classes)
    results.update({k : accuracy})

plot_barchart(results, 0, 400, 0.9, 0.98)
train_data_new = read_arff("../input/train-skewed.arff")
train_data_new = np.array(train_data_new)
print("datapoints of train data:", len(train_data))

test_data_new = pd.read_csv("../input/test-skewed.csv", index_col=0, header=0, names=['Id (String)', 'X', 'Y']) # read the test data
test_data_new = np.array(test_data_new[['X','Y']].values) # put data in numpy array
print("datapoints of test set:", len(test_data_new))

train_input_new = train_data_new[:, 0:2]
train_classes_new = train_data_new[:, 2]
plot_datapoints(train_data_new)
results = {}

for k in range(1, 399):
    classifier = neighbors.KNeighborsClassifier(k)
    classifier.fit(train_input_new, train_classes_new)  # train the classifier
    predicted_classes = classifier.predict(train_data_new[:,0:2])  # let it run over the train data
    # measure and save the accuracy 
    accuracy = true_accuracy(predicted_classes)
    results.update({k : accuracy})

plot_barchart(results, 0, 400, 0, 1)
from sklearn import model_selection
results = {}

for k in range(1, math.floor(399 * 0.6)):
    classifier = neighbors.KNeighborsClassifier(k)
    cv_split = model_selection.ShuffleSplit(10, 0.3, 0.6, 0)
    cv_results = model_selection.cross_validate(classifier, train_input_new, train_classes_new, cv=cv_split) 
    score = cv_results["test_score"].mean()
    results.update({k : score})
    
plot_barchart(results, 0, math.floor(399 * 0.6), 0, 1)
plot_barchart(results, 0, 40, 0.9, 1)
classifier = neighbors.KNeighborsClassifier(8)
classifier.fit(train_input_new, train_classes_new)
predicted_classes = classifier.predict(test_data_new)
accuracy = true_accuracy(predicted_classes)
print(accuracy)
results = {}

for k in range(1, 399):
    classifier = neighbors.KNeighborsClassifier(k)
    classifier.fit(train_input_new, train_classes_new)  # train the classifier
    predicted_classes = classifier.predict(test_data_new)  # let it run over the train data
    # measure and save the accuracy 
    accuracy = true_accuracy(predicted_classes)
    results.update({k : accuracy})

plot_barchart(results, 0, 400, 0, 1)
plot_barchart(results, 0, 40, 0.9, 1)