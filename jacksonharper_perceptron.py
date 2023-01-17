import csv

import math

import random



# Load CSV file, remove headers if any are dfound, 

# convert values to floats, and remove rows that 

# don't have x and y values

def load_csv(filename):

    dataset = list()

    with open(filename) as csvfile:

        reader = csv.reader(csvfile)

        for row in reader:

            for i in range(len(row) - 1):

                row[i] = float(row[i].strip())

            row[-1] = 1.0 if row[-1] == 'R' else 0.0

            dataset.append(row)

    return dataset



def train_test_split(dataset, split):

    train = list()

    test = list(dataset)

    train_len = len(dataset) * split

    

    while len(train) < train_len:

        index = random.randrange(len(test))

        item = test.pop(index)

        train.append(item)

    return train, test



train, test = train_test_split(load_csv('../input/sonar.all-data.csv'), 0.6)



print('train %s . test . %s' % (len(train), len(test)))

def predict(row, weights):

    activation = weights[0]

    for i in range(len(row) - 1):

        activation += weights[i + 1] * row[i]

    return 1.0 if activation >= 0.0 else 0.0



def train_weights(train, l_rate, n_epoch):

    weights = [0.0] * len(train[0])

    for epoch in range(n_epoch):

        sum_error = 0.0

        for row in train:

            prediction = predict(row, weights)

            error = row[-1] - prediction

            sum_error = sum_error + error ** 2

            weights[0] = weights[0] + l_rate * error

            for i in range(len(row) - 1):

                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]

    return weights



def perceptron(train, test, l_rate, n_epoch):

    predictions = list()

    weights = train_weights(train, l_rate, n_epoch)

    for row in test:

        prediction = predict(row, weights)

        predictions.append(prediction)

    return predictions



def stripped_test_set_and_actual(dataset):

    test = list()

    actual = list()

    for row in dataset:

        c = list(row)

        actual.append(c[-1])

        c[-1] = None

        test.append(c)

    return test, actual



def accuracy_metric(actual, predicted):

    correct = 0

    for i in range(len(actual)):

        if actual[i] == predicted[i]:

            correct = correct + 1

    return correct / len(actual) * 100



def evaluate(train, test, algorithm, *args):

    test_set, actual = stripped_test_set_and_actual(test)

    predicted = algorithm(train, test, *args)

    accuracy = accuracy_metric(actual, predicted)

    return accuracy





accuracy = evaluate(train, test, perceptron, 0.01, 50)

print('accuracy: ', accuracy)