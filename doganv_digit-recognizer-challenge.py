import numpy as np



class NeuralNetwork:

    def __init__(self, amount_of_inputs, amount_of_hidden_layers, amount_of_nodes_per_layer, amount_of_outputs):

        self.amount_of_outputs = amount_of_outputs

        self.layers = []

        self.layers.append(FirstHiddenLayer(amount_of_inputs, amount_of_nodes_per_layer))

        for i in range(amount_of_hidden_layers - 1):

            self.layers.append(Layer(amount_of_nodes_per_layer))

        self.layers = np.array(self.layers)

        self.output_layer = OutputLayer(amount_of_nodes_per_layer)



    # inputs = list

    def feed_forward(self, inputs):

        for i in range(self.layers.shape[0]):

            inputs = np.ndarray.tolist(self.layers[i].calc_outputs(inputs))

        return inputs



    def get_guess(self, inputs):

        forwarded_inputs = self.feed_forward(inputs)

        output = self.output_layer.calc_outputs(forwarded_inputs)

        guess = guess_func(output, self.amount_of_outputs)

        return guess



    def get_error(self, inputs, correct_result):

        return correct_result - self.get_guess(inputs)



    def __learn(self, inputs, correct_result):

        inputs = inputs + [1]

        error = self.get_error(inputs, correct_result)

        if error != 0:

            no_errors = False

            for i in range(self.layers.shape[0]):

                error = self.layers[self.layers.shape[0] - i - 1].update_weights(error)

        else:

            no_errors = True

        return no_errors



    def learn_limited_amount_of_times(self, inputs, correct_result, amount_of_tries):

        not_done = True

        for i in range(amount_of_tries):

            if self.__learn(inputs, correct_result):

                not_done = False

                break

        return not_done



    def learn_with_multiple_inputs(self, inputs, correct_results, amount_of_tries):

        for i in range(amount_of_tries):

            for j in range(len(inputs)):

                self.learn_limited_amount_of_times(inputs[j], correct_results[j], amount_of_tries)
import numpy as np



class Layer:

    def __init__(self, amount_of_perceptrons):

        self.amount_of_perceptrons = amount_of_perceptrons

        self.weights = self.generate_weights(amount_of_perceptrons, amount_of_perceptrons)

        self.learning_rate = 1.0 / amount_of_perceptrons

        self.activation_function = np.vectorize(self.activate, otypes=[np.float])  # select activation function



    def calc_outputs(self, inputs):

        return self.activate_all_elements_of_matrix(np.dot(self.weights, inputs))



    def activate_all_elements_of_matrix(self, inputs):

        return self.activation_function(inputs)



    @staticmethod

    def activate(inputs):  # sigmoid

        return 1 / (1 + np.exp(-inputs))



    # returns matrix with amountOfPerceptrons = rows, amountOfWeights (inputs) = columns

    # and with random values between -1 and 1 (with two digits after the .)

    @staticmethod

    def generate_weights(amount_of_inputs, amount_of_perceptrons):

        weights = np.random.randint(-100, 100, (amount_of_perceptrons, amount_of_perceptrons)) / 100.0

        weights[weights == 0] = 0.1

        return weights



    # def sums_of_weights(self):

    #     sums = list()

    #     for i in range(self.weights.shape[0]):

    #         sum = 0

    #         for j in range(self.weights.shape[1]):

    #             sum += self.weights[i][j]

    #         sums.append(sum)

    #     return sums if sums != 0 else 0.01



    # returns errors for next layer

    def update_weights(self, error):

        # sums = self.sums_of_weights()

        output_error = list()

        if isinstance(error, list):

            error_len = len(error)

        else:

            error_len = 1

        # for i in range(len(sums)):

        for i in range(error_len):

            one_error = list()  # naming is bad

            for j in range(self.weights.shape[1]):

                # if sums[i] == 0:

                #     sums[i] = 0.01

                if isinstance(error, list):

                    one_error.append(error[i] * self.weights[i, j] / 100)  # / sums[i])

                else:

                    one_error.append(error * self.weights[i, j] / 100)  # / sums[i])

                self.weights[i, j] += one_error[j] * self.learning_rate

            output_error.append(sum(one_error) / len(one_error))

        return output_error
class OutputLayer(Layer):

    def __init__(self, amount_of_perceptrons):

        Layer.__init__(self, amount_of_perceptrons)

        self.weights = self.generate_weights(1, amount_of_perceptrons)



    @staticmethod

    def generate_weights(amount_of_perceptrons, amount_of_inputs):

        return np.random.randint(-100, 100, (amount_of_perceptrons, amount_of_inputs)) / 100.0





class FirstHiddenLayer(Layer):

    def __init__(self, amount_of_inputs, amount_of_perceptrons):

        Layer.__init__(self, amount_of_perceptrons)

        self.weights = self.generate_weights(amount_of_inputs + 1, amount_of_perceptrons)  # +1 for the bias



    @staticmethod

    def generate_weights(amount_of_perceptrons, amount_of_inputs):

        return np.random.randint(-100, 100, (amount_of_inputs, amount_of_perceptrons)) / 100.0
def guess_func(value, amount_of_outputs):

    return int((value + 0.5) * amount_of_outputs % amount_of_outputs)
import pandas as pd

import numpy as np



#variables

amount_of_hidden_layers = 5  # how many hidden layers are used

amount_of_neurons_per_layer = 100  # how many neurons per hidden layer are used

amount_of_different_outputs = 10  # how many possible outputs there are at the end (the amount of different classifictaiosn)

amount_of_used_learning_samples = -1  # how many of the learning samples are used (if <= 0 or >= len(inputs) then all)

amount_of_usages_per_learning_sample = 5  # max amount of tries to get the correct result for each learning sample



# open test datas

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



# save correct results

answers = train["label"]



# remove "label" column

inputs = train.drop(labels=["label"], axis=1)



# normalize the values

inputs = inputs / 255.0

test = test / 255.0



# not needed anymore

del train



# create Neural Network

neural_network = NeuralNetwork(784, amount_of_hidden_layers, amount_of_neurons_per_layer, amount_of_different_outputs)



import random



if amount_of_used_learning_samples <= 0 or amount_of_used_learning_samples >= len(inputs):

    # uses all learning samples in a random order

    indexes = list(range(len(inputs)))

    random.shuffle(indexes)

    progress = 0

    for i in indexes:

        print("%sth testdata" % progress)

        progress += 1

        one_input = inputs.loc[i, :].values

        answer = answers.loc[i]

        one_input = np.ndarray.tolist(one_input)

        neural_network.learn_limited_amount_of_times(one_input, answer, amount_of_usages_per_learning_sample)

else:

    # uses the given amount of random learning samples

    for i in range(amount_of_used_learning_samples):

        print("%sth testdata" % i)

        # choose a random test data

        random_int = random.randint(0, len(inputs) - 1)

        one_input = inputs.loc[random_int, :].values

        answer = answers.loc[random_int]

        one_input = np.ndarray.tolist(one_input)

        neural_network.learn_limited_amount_of_times(one_input, answer, amount_of_usages_per_learning_sample)



import csv



# create csv and save the results in it

with open('test_submission.csv', 'w', newline='') as test_submission:

    writer = csv.writer(test_submission)

    # add the header

    writer.writerow(['ImageId', 'Label'])



    # go through all test datas in test.csv

    for i in range(len(test)):

        print("submission: %s" % i)

        guess = neural_network.get_guess(np.ndarray.tolist(test.loc[i, :].values) + [1])

        # write result in the output file

        writer.writerow([(i + 1), guess])



test_submission.close()

print("done")
