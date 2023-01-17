import numpy as np

class MultilayerPerceptron:



    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):

        self.inNodes = inputNodes

        self.hidNodes = hiddenNodes

        self.outNodes = outputNodes

        self.lR = learningRate

        #Zufällige Gewichte in der Region [-0.5;0,5)

        self.weightsIH = (np.random.rand(self.hidNodes, self.inNodes) - 0.5)

        self.weightsHO = (np.random.rand(self.outNodes, self.hidNodes) - 0.5) 

        

    def sigmoid(self, x):

        return 1 / (1 + np.exp(-x))

        

        

    def forwardPropagation(self, input_list):

        inputs = np.array(input_list, ndmin=2)

        hiddenInputs = np.dot(self.weightsIH, inputs.T)

        hiddenOutputs = self.sigmoid(hiddenInputs)

        finalInputs = np.dot(self.weightsHO, hiddenOutputs)

        finalOutputs = self.sigmoid(finalInputs)

        return hiddenInputs, hiddenOutputs, finalInputs, finalOutputs

    

    def backPropagation(self, input_list, targets_list):

        inputs = numpy.array(input_list, ndmin=2).T

        targets = numpy.array(targets_list, ndmin=2).T

        

        hiddenInputs, hiddenOutputs, finalInputs, finalOutputs = self.forwardPropagation(input_list)

        

        outputErrors = targets - finalOutputs

        hiddenErrors = np.dot(self.weightsHO.T, outputErrors)

        

        self.weightsIH += self.lR * np.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), inputs.T)

        self.weightsHO += self.lR * np.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), hiddenOutputs.T)

        

        

    def test(self, inputs):

        a,b,c,d = self.forwardPropagation(inputs)

        return d



    

        
import numpy

# Anzahl der Knoten auf den verschiedenen Schichten

input_nodes = 784

hidden_nodes = 100

output_nodes = 10



generationen = 12



# Lernrate

learning_rate = 0.3



n = MultilayerPerceptron(input_nodes, hidden_nodes, output_nodes, learning_rate)



# Laden der mnist Trainigs Daten in eine Liste

training_data_file = open("../input/train.csv", "r")

training_data_list = training_data_file.readlines()

training_data_file.close()



training_data_list = training_data_list[1:]



for i in range(generationen):

    # Trainieren des Neuronales Netzes

    for eintrag in training_data_list:

        # An den Kommas splitten

        all_values = eintrag.split(',')

        # Eingaben Skalieren

        scaled_input = (numpy.asfarray(all_values[1:]) / 255 * 0.99) + 0.01

        # Zielwert-Array

        targets = numpy.zeros(output_nodes) + 0.01

        # Kennung des Untersuchten Eintrags wird in Zielarray gespeichert

        targets[int(all_values[0])] = 0.99

        n.backPropagation(scaled_input, targets)

# MLP wird abgefragt und Ergebnisse werden in eine csv-Datei gespeichert

import csv



header = [['ImageId', 'Label']]



with open('submission.csv', 'w') as csvFile:

    writer = csv.writer(csvFile)

    writer.writerows(header)

test_data_file = open("../input/test.csv", "r")

test_data_list = test_data_file.readlines()

test_data_file.close()



test_data_list = test_data_list[1:]



def maximum(array):

    größterEintrag = 0

    index = 0

    i = 0

    for eintrag in array:

        if eintrag[0] > größterEintrag:

            größterEintrag = eintrag[0]

            index = i

        i += 1

    return index



index = 1

for eintrag in test_data_list:

    all_values = eintrag.split(',')

    scaled_input = (numpy.asfarray(all_values) / 255 * 0.99) + 0.01

    row = [index, maximum(n.test(scaled_input))]

    with open('submission.csv', 'a') as csvFile:

        writer = csv.writer(csvFile)

        writer.writerow(row)

    index += 1

csvFile.close()
