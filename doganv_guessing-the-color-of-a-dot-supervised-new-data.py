from prettytable import PrettyTable

import time



class Final_result:

    result = []

    header = []

    

    def __init__(self, header):

        self.header = header

        

    def __format_accuracy(self, value):

        # turn it into a percentage number and correctly round by 2nd decimal digit

        return str(int(value * 10000 + 0.5) / 100.0) + "%"

        

    def add_entry(self, name_of_solution, results_of_solution, needed_time):

        output = [name_of_solution]

        accuracies = []

        for i in range(len(results_of_solution)):

            accuracy = compare_with_true_result(results_of_solution[i], i)

            output.append(self.__format_accuracy(accuracy))

            accuracies.append(accuracy)

        time_in_ms = needed_time * 1000

        rounded_time = int(time_in_ms * 10 + 0.5) / 10

        output.append("%s ms" % rounded_time)

        output.append(self.score(accuracies, needed_time))

        self.result.append(output)

        

    def score(self, accuracies, needed_time):

        sum = 0

        for accuracy in accuracies:

            sum += accuracy

        avg_accuracy = sum / len(accuracies)

        # score = accuracy * fifth_root_of(1 / time)

        score = avg_accuracy * ((1.0 / needed_time) ** (1.0 / 5))

        rounded_score = int(score * 100 + 0.5) / 100

        return rounded_score

    

    def change_header(self, header):

        self.header = header

        

    def print_as_table(self):

        quicksort(self.result, -1)

        self.result.reverse()

        pretty_table = PrettyTable(field_names=self.header)

        for row in self.result:

            pretty_table.add_row(row)

        print(pretty_table.get_string())
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
training_data = []

training_data.append(read_data("../input/train.arff"))

test = read_data("../input/eval.arff")

new_test = []

correct_result = [[], []]

for data in test:

    new_test.append([data[0], data[1]])

    correct_result[0].append(data[2])

test = []

test.append(new_test)

del new_test
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv('../input/train.csv')

np_train_X = train_data[['X','Y']].values

np_train_Y = train_data[['class']].values



test_data = pd.read_csv('../input/test.csv')

np_test_X = test_data[['X','Y']].values

np_test_Y = test_data[['class']].values

import matplotlib.pyplot as plt



def show_data(X,data):

    colors = {0:'red',1:'blue'}

    plt.figure(figsize=(5,5))

    plt.scatter(X[:,0],X[:,1],c=data["class"].apply(lambda x: colors[x]))

    plt.xlabel("x")

    plt.ylabel("y")

    plt.show()



show_data(np_train_X,train_data)

show_data(np_test_X,test_data)

print(np.array(np_train_X).shape)

print(np.array(np_train_Y).shape)

# merging data and result in training_data to get the same format as before

train_data_2 = np.ndarray.tolist(np_train_X)

for i in range(len(np_train_X)):

    train_data_2[i].append(np_train_Y[i][0])

training_data.append(train_data_2)

list_test_X = np.ndarray.tolist(np_test_X)

list_test_Y = np.ndarray.tolist(np_test_Y)

test_1 = []

for i in range(len(list_test_X)):

    test_1.append(list_test_X[i])

test.append(test_1)

for i in range(len(list_test_Y)):

    correct_result[1].append(list_test_Y[i])

for i in range(len(correct_result[1])):

    correct_result[1][i] = correct_result[1][i][0]

print(np.array(np_test_X).shape)

print(np.array(np_test_Y).shape)
output_header = ["name_of_solution", "triangle_data", "circle_data", "time", "score"]

final_output = Final_result(output_header)

# constants to increase readebility

X = 0

Y = 1

COLOR = 2

CLASSES = [[1, -1], [1, 0]]  # should go through the data to find this out

# saving .1MB RAM with these dels

del train_data_2

del np_test_X

del np_test_Y

del test_data

del train_data

del np_train_X

del np_train_Y

del output_header
def compare_with_true_result(values, pos):

    i = 0

    correct = 0

    while i < len(correct_result[pos]):

        if correct_result[pos][i] == values[i]:

            correct += 1

        i += 1

    return correct / len(correct_result[pos])
#linear function

def f(x):

    return -x



def __get_guess(x, y, classes):

    return classes[0] if y >= f(x) else classes[1]



def get_guess(values, classes):

    return __get_guess(values[0], values[1], classes)
start_time = time.perf_counter()

my_result = []

for n in range(len(test)):  # iterating through the amount of different cases

    my_result.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(get_guess(test[n][i], CLASSES[n]))

    print("1. solving the task with a linear function: %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "linear function"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
# the other function is the negative version of this one

def quadratic_function(x):

    return 1/0.6 * (x ** 2) - 0.6



def __between_functions(x, y, classes):

    is_between = False

    if y >= quadratic_function(x) and y <= -quadratic_function(x):

        is_between = True

    return is_between

    

def between_functions(unknown_dot, classes):

    return __between_functions(unknown_dot[X], unknown_dot[Y], classes)

    

def get_guess(unknown_dot, classes):

    is_between = between_functions(unknown_dot, classes)

    if is_between:

        guess = classes[0]

    else:

        guess = classes[1]

    return guess
start_time = time.perf_counter()

my_result = []

for n in range(len(test)):  # iterating through the amount of different cases

    my_result.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(get_guess(test[n][i], CLASSES[n]))

    print("1.1. solving the task with two quadratic functions: %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "quadratic functions"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
def partition(array, pos, begin, end):

    pivot = begin

    for i in range(begin+1, end+1):

        if array[i][pos] <= array[begin][pos]:  # changed this line so that it sorts by X or Y value

            pivot += 1

            array[i], array[pivot] = array[pivot], array[i]

    array[pivot], array[begin] = array[begin], array[pivot]

    return pivot





def quicksort(array, pos, begin=0, end=None):

    if end is None:

        end = len(array) - 1

    def _quicksort(array, pos, begin, end):

        if begin >= end:

            return

        pivot = partition(array, pos, begin, end)

        _quicksort(array, pos, begin, pivot-1)

        _quicksort(array, pos, pivot+1, end)

    return _quicksort(array, pos, begin, end)
def get_closest_color(unknown_dot, training_data):

    no_dots_in_range = True

    current_square_size = 0

    #

    # training_data is already sorted by X

    square_size = (training_data[-1][X] - training_data[0][X]) * 0.1

    #

    # counter = 0  # just for the print

    training_data_only_in_range = []

    while no_dots_in_range:

        current_square_size += square_size

        training_data_only_in_range = remove_unneeded_X_and_Y_data(unknown_dot, current_square_size, training_data)

        no_dots_in_range = not training_data_only_in_range  # checks if the list is empty

        # if no_dots_in_range:

        #     print("no dots in range %s" % counter)

        #     counter += 1

    return training_data_only_in_range[index_of_closest_dot(unknown_dot, training_data_only_in_range)][COLOR]

    

def remove_unneeded_X_and_Y_data(unknown_dot, square_size, training_data):

    for pos in range(2):  # 2 because we are using 2 dimensional data (x and y)

        if pos != X:  # the given one is already sorted by X  --  this almost halves the execution time

            quicksort(training_data, pos)  # sorting the data by X or Y according to the value of pos

        index = get_first_and_last_index_in_range(training_data, unknown_dot, square_size, pos)

        first_index_in_range = index[0]

        first_index_after_range = index[1]

        training_data = training_data[first_index_in_range : first_index_after_range]

    return training_data



# pos == 0 => X

# pos == 1 => Y

def get_first_and_last_index_in_range(data, unknown_dot, square_size, pos):

    first_index_in_range = -1

    first_index_after_range = -1

    i = 0

    while i < len(data) and first_index_after_range == -1:

        if first_index_in_range == -1:

            if data[i][pos] >= unknown_dot[pos] - square_size:

                first_index_in_range = i

        elif first_index_after_range == -1 and data[i][pos] > unknown_dot[pos] + square_size:

            first_index_after_range = i

        i += 1

    if first_index_in_range == -1:  # true if the data is too far at the left

        first_index_in_range = 0

    if first_index_after_range == -1:  # true if the data is too far at the right

        first_index_after_range = len(data) - 1

    return [first_index_in_range, first_index_after_range]



def index_of_closest_dot(unknown_dot, training_data):

    distance = 999999

    index = -1

    for i in range(len(training_data)):

        new_distance = distance_between_dots(training_data[i], unknown_dot)

        if new_distance < distance:

            distance = new_distance

            index = i

    return index



def distance_between_dots(dot1, dot2):

    # to get the actual distance you would need to take the square root of it but that doesn't matter

    # because we never take the square root so it is consistent

    return (dot1[X] - dot2[X]) ** 2 + (dot1[Y] - dot2[Y]) ** 2
start_time = time.perf_counter()

my_result = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    my_result.append([])

    training_data_n_sorted_by_X = training_data[n]

    quicksort(training_data_n_sorted_by_X, X)

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(get_closest_color([test[n][i][X], test[n][i][Y]], training_data_n_sorted_by_X))

    print("2. nearest neighbor (aprox): %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "nearest neighbor (aprox)"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
def get_closest_color(unknown_dot, training_data):

    closest_distance_and_color = (999999, 1)

    for data in training_data:

        new_distance_and_color = (distance_between_dots(unknown_dot, data), data[COLOR])  # using the distance_between_dots() from the one version before

        if new_distance_and_color[0] < closest_distance_and_color[0]:  # compare distances

            closest_distance_and_color = new_distance_and_color

    return closest_distance_and_color[1]  # return only the color
start_time = time.perf_counter()

my_result = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    my_result.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(get_closest_color([test[n][i][X], test[n][i][Y]], training_data[n]))

    print("2.1. nearest neighbour (expo): %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "nearest neighbour (expo)"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
import random

def sign(value):

    if value >= 0:

        value = 1

    else:

        value = -1

    return value



class PerceptronIO:

    def __init__(self, inputs, correctValue):

        self.inputs = inputs + [1]  # +1 for the bias

        self.correctValue = correctValue



class Perceptron:

    def __init__(self, perceptronIOs, learningRate, amountOfTries):

        self.perceptronIOs = perceptronIOs

        self.weights = list()

        self.generateWeights()

        self.learningRate = learningRate

        self.amountOfTries = amountOfTries



    def generateWeights(self):  # 1 per input

        for i in range(len(self.perceptronIOs[0].inputs)):

            self.weights.append(random.randrange(-100, 100, 1) / 100)



    def sum(self, perceptronIO):  # 1 per weight

        sum = 0

        for i in range(len(self.weights)):

            sum += self.weights[i] * perceptronIO.inputs[i]

        return sum



    def sums(self):  # 1 per IO

        sums = list()

        for i in range(len(self.perceptronIOs)):

            sums.append(self.sum(self.perceptronIOs[i]))

        return sums

    

    def guesses(self):  # 1 per IO

        guesses = list()

        sums = self.sums()

        for i in range(len(sums)):

            guesses.append(sign(sums[i]))

        return guesses



   # def error(self):

   #     return correctValue - guess

    def errors(self):  # 1 per IO

        errors = list()

        guesses = self.guesses()

        for i in range(len(self.perceptronIOs)):

            errors.append(self.perceptronIOs[i].correctValue - guesses[i])

        return errors



    def adjustWeights(self):  # returns whether the weightsChanged or not

        errors = self.errors()

        #create a copy of the oldWeights to adjust them without manipulating the values for the calculations

        newWeights = list()

        for i in range(len(self.weights)):

            newWeights.append(self.weights[i])



        for i in range(len(self.perceptronIOs)):  # weight += error * input * learning rate

            for c in range(len(self.weights)):

                newWeights[c] += errors[i] * self.perceptronIOs[i].inputs[c] * self.learningRate

        weightsChanged = self.weights != newWeights

        self.weights = newWeights

        return weightsChanged



    def learn(self):

        i = 0

        while self.adjustWeights() and i < self.amountOfTries:

            i += 1

        print("done training")

                

    #output for a specific case

    def specificGuess(self, x, y):

        return sign(self.sum(PerceptronIO([x, y], -1)))
start_time = time.perf_counter()

learning_rate = 0.1  # 1 / len(perceptron_inputs[0])

amount_of_tries = 100

perceptron = []

perceptron_inputs = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    perceptron_inputs.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        perceptron_inputs[n].append(PerceptronIO([training_data[n][i][0], training_data[n][i][1]]

                                                 , training_data[n][i][2]))

    perceptron.append(Perceptron(perceptron_inputs[n], learning_rate, amount_of_tries))

    perceptron[n].learn()
my_result = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    my_result.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(perceptron[n].specificGuess(test[n][i][0], test[n][i][1]))

    print("3. single perceptron: %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "single perceptron"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
from sklearn.neural_network import MLPClassifier



mlp = []

training_data_inputs = []

training_data_outputs = []

for n in range(len(training_data)):

    mlp.append(MLPClassifier(max_iter=5000))

    training_data_inputs.append([])

    training_data_outputs.append([])

    for i in range(len(training_data[n])):

        training_data_inputs[n].append([training_data[n][i][X], training_data[n][i][Y]])

        training_data_outputs[n].append(training_data[n][i][COLOR])

    mlp[n].fit(training_data_inputs[n], np.ravel(training_data_outputs[n],order='C'))  # training_data_outputs[n])
start_time = time.perf_counter()

my_result = []

results = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    my_result.append([])

    results.append(mlp[n].predict(test[n]))

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(results[n][i])

    print("4. MLP of a library: %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "sklearn MLP"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
# should find the most common color instead of just using one

start_time = time.perf_counter()

my_result = []

for n in range(len(test)):  # iterating through the amount of different cases (circle and triangle data)

    my_result.append([])

    for i in range(len(test[n])):  # iterating through all given test data

        my_result[n].append(CLASSES[n][1])

    print("5. assuming the same color for all the dots: %s" % compare_with_true_result(my_result[n], n))

name_of_this_solution = "all same color"

needed_time = time.perf_counter() - start_time

final_output.add_entry(name_of_this_solution, my_result, needed_time)
final_output.print_as_table()