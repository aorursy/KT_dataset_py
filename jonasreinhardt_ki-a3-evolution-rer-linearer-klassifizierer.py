import numpy as np
import math
import datetime as dt
import random as rd
import pandas as pd
import csv

def read_data(filename: str):
    """reads the file and puts the data into a 2-dimensional list"""
    f = open(filename)
    data_line = False
    data = []
    for l in f:
        l = l.strip()  # get rid of newline at the end of each input line
        if data_line:
            content = [float(x) for x in l.split(',')]
            if len(content) == 3:
                data.append(content)
        else:
            if l.startswith('@DATA'):
                data_line = True
    return data
def create_trainset(size: int, data, random=True):
    """creates both train and test-data by randomly splitting the data into two lists.

    :arg size: how many values should be put into the tain-data. all unused values will be put into the test-data.
    :arg data: the data to be split
    :arg random: whether the split of the train-set between train- and self-evaluation-set should be the same for
    every generated classifier or different (random) for every generated classifier"""
    if random:
        data_temp = data.copy()
        rd.shuffle(data_temp)
        traindata = data_temp[0:size - 1]
        testdata = data_temp[size:len(data_temp)]
        return [traindata, testdata]
    else:
        data_temp = data.copy()
        traindata = data_temp[0:size - 1]
        testdata = data_temp[size:len(data_temp)]
        return [traindata, testdata]
class LinearClassifier:
    """classifies data with 2-dimensional input according to 2 possible classes (in this case 1 or -1)"""

    # the data this classifier is allowed to use to train
    train = []
    # the data this classifier is allowed to use to evaluate itself
    test = []

    # the default values for the linear function (f(x) = mx + n) of this classifier
    m_def = 0
    n_def = 0

    def __init__(self, train, test, m, n):
        self.train = train
        self.test = test
        self.m_def = m
        self.n_def = n

    def classify_data(self, data, m=m_def, n=m_def):
        """classifies a set of data according to a linear function"""
        result = []
        for date in data:
            x = date[0]
            y = date[1]
            result.append([x, y, self.classify(x, y, m, n)])
        return result

    def classify(self, x, y, m, n):
        """classifies exactly one unit of data according to the given linear function. All points with y > mx + b recieve the class '1',
        all points with y <= mx + b recieve the class '-1'"""
        return 1 if y >= self.linear_func(x, m, n) else -1

    def linear_func(self, x, m, n):
        """returns the value y with y = f(x) = mx + n"""
        return m * x + n

    def evaluate(self, test, goal):
        """evaluates the classifier by comparing classified test-data against the same data with the 'correct' values"""
        amnt_correct = 0
        for i in range(len(test)):
            if test[i][2] == goal[i][2]:
                amnt_correct += 1

        return amnt_correct / len(test)

    def train_data(self, interval_size=len(train), epsilon=0.05, randomness=0.0):
        """trains the classifier along a set of train-data, according to the given parameters. This is done by deducing
        a set of new functions from the last default-function by changing the parameters m (slope) and n (y-intercept).
        The new functions are deduced  decreasing and increasing m aswell as n by respectively m +- epsilon*m and
        n +- epsilon*n. The randomness-factor then randomly changes m and n by adding
        (np.random.random() - 0.5) * randomness to m an n respecitvely.

        then all the new functions are tested against a subset of the train-data. The function that performs best will
        become the new default-function and the reference for the next iteration of training.

        The amount of iterations is provided by the interval_size argument, which sets the size of each subset. e.g.:
        the train-data has a size of 500, and the interva_size is set to 100, then the function will perform 5
        iterations and thus 5 new generations of functions."""

        index_start = 0  # remeber the start-index of the slice of the current iteration in the train-data.

        # each loop represents a iteration over the test-data-interval and will produce a new generation of the function
        for i in range(int(math.floor(len(self.train) / interval_size))):
            current_slice = self.train[index_start:index_start + interval_size]

            # a dictionary that maps all the newly-calculated values for m and n to their sucess-rate
            current_evaluation = {}

            # enumerating all possible new values for m and n
            m_values = [self.m_def,
                        self.m_def + ((np.random.random() - 0.5) * randomness),
                        self.m_def - epsilon,
                        self.m_def - epsilon + ((np.random.random() - 0.5) * randomness),
                        self.m_def + epsilon,
                        self.m_def + epsilon + ((np.random.random() - 0.5) * randomness)]
            n_values = [self.n_def,
                        self.n_def + ((np.random.random() - 0.5) * randomness),
                        self.n_def - epsilon,
                        self.n_def - epsilon + ((np.random.random() - 0.5) * randomness),
                        self.n_def + epsilon,
                        self.n_def + epsilon + ((np.random.random() - 0.5) * randomness)]

            # for each combination of m and n, evaluate their success rate among to the current slice of test-data
            for m in m_values:
                for n in n_values:
                    results = self.classify_data(current_slice, m, n)
                    eval = {(m, n): self.evaluate(current_slice, results)}
                    current_evaluation.update(eval)

            # sort the dictionary acccording to the value (the success rate)
            current_evaluation = sorted(current_evaluation.items(), key=lambda x: x[1])

            # at the end, assing the 'best' m and n as the new default values for m and n for this classifier
            self.m_def, self.n_def = current_evaluation.pop(len(current_evaluation) - 1)[0]

            # update the index to mark the start of the next slice
            index_start += interval_size

    def train_and_evaluate(self, data, classifier_amnt, size_train, size_train_interval, epsilon, randomness,
                           display_amnt=10, m_default=0, n_default=0, random_train_set=False):
        """Raising the evolutionary approach to a second level, this method will create a certain amount of classifier-
        objects and test them against the same set of data. The classifiers that perform the best will be printed out
        and the best one will be returned
        :arg data: The data the classifiers will use to train
        :arg classifier_amnt: the amount of classifiers to be generated
        :arg size_train: the size of the train-set. Must be smaller than data
        :arg size_train_interval: the interval size of each iteration of training
        :arg epsilon: the epsilon value to mutate the function
        :arg randomness: the factor of randomness to further, randomly mutate the function
        :arg display_amnt: the amound of classifiers that will be displayed in the end
        :arg m_default: the default value for m
        :arg n_default: the default value for n"""
        current_time = dt.datetime.now()
        classifiers = {}
        best_classifier = None
        for i in range(classifier_amnt):
            traindata, testdata = create_trainset(size_train, data, random_train_set)

            linear_classifier = LinearClassifier(traindata, testdata, m_default, n_default)
            size_test = len(testdata)
            linear_classifier.train_data(size_train_interval, epsilon, randomness)
            classified_data = linear_classifier.classify_data(linear_classifier.test, linear_classifier.m_def,
                                                              linear_classifier.n_def)
            classifiers.update(
                {linear_classifier: linear_classifier.evaluate(classified_data, linear_classifier.test)})

        classifiers = sorted(classifiers.items(), key=lambda x: x[1])
        delta_time = dt.datetime.now() - current_time

        print("A total of ", size_train + size_test, " data points have been used, ", size_train,
              " to train the classifier and ", size_test, " to evaluate the classifier.\nA total of ",
              classifier_amnt,
              " classifiers have been created individually.\nThe sample size for each train iteration was",
              size_train_interval, ","
                                   " epsilon was ", epsilon, " and the factor of randomness was ", randomness,
              ".\nIt took exactly ", delta_time, " to finish the process.")

        print("printing the best ", display_amnt, " classifiers:")

        for c in reversed(classifiers):
            if best_classifier == None: best_classifier = c[0]
            print("Function of linear classifier: f(x) = ", c[0].m_def, "x + ", c[0].n_def, " with a sucess rate of ",
                  c[1])
            display_amnt -= 1
            if display_amnt == 0:
                break
        return best_classifier

data = read_data("../input/kiwhs-comp-1-complete/train.arff")
m_default = -1
n_default = -0.5
#Die Werte sind hier so gewählt, wie sie bereits halbwegs sinnvoll klassifizieren. Man könnte aber auch mit m,n = 0 initialisieren
version_id = 16
random_train_set = True
c = LinearClassifier(None, None, None, None) #der Klassifizierer, der Instanzen von sich selbst generiert und evaluert
size_train = 300
size_train_iteration = 100
epsilon = 0.05
randomness = 0.1
classifier_amnt = 10000
display_amnt = 10
best_classifier = c.train_and_evaluate(data, classifier_amnt, size_train, size_train_iteration, epsilon, randomness, display_amnt, m_default, n_default, random_train_set)
with open("../input/kiwhs-comp-1-complete/test.csv") as f:
    reader = csv.reader(f)
    test = list(reader)
test.pop(0) #column labels not needed for classification
for i in test:
    i.pop(0) #ID not needed for classification
    i[0] = float(i[0])
    i[1] = float(i[1])
    
result = best_classifier.classify_data(test) #classify the data

index = 0
for i in result:
    i.pop(0)
    i.pop(0) #leave only the class, x,y coordinates not needed
    i.insert(0, index)
    index += 1
#print(result)
result.insert(0, ["Id (String)","Category (String)"])
with open("ouput" + str(version_id) + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(result)
