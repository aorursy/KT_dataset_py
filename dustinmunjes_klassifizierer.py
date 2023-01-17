# Read some arff data



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



# Aufbereiten der Trainings- und Evaluationsdaten



daten = read_data("../input/trainingsdaten/trainingsdaten.txt")

zielwerte = []

eingabewerte = []

for wert in daten:

    eingabewerte.append(wert[:-1])

    if wert[2] < 0:

        zielwerte.append([0.99, 0])

    else:

        zielwerte.append([0, 0.99])



evaldaten = read_data("../input/evaluationsdaten/evaluationsdaten.txt")

evalzielwerte = []

evaleingabewerte = []

for wert in evaldaten:

    evaleingabewerte.append(wert[:-1])

    if wert[2] < 0:

        evalzielwerte.append([0.99, 0])

    else:

        evalzielwerte.append([0, 0.99])

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

        inputs = np.array(input_list, ndmin=2).T

        targets = np.array(targets_list, ndmin=2).T

        

        hiddenInputs, hiddenOutputs, finalInputs, finalOutputs = self.forwardPropagation(input_list)

        

        outputErrors = targets - finalOutputs

        hiddenErrors = np.dot(self.weightsHO.T, outputErrors)

        

        self.weightsIH += self.lR * np.dot((hiddenErrors * hiddenOutputs * (1.0 - hiddenOutputs)), inputs.T)

        self.weightsHO += self.lR * np.dot((outputErrors * finalOutputs * (1.0 - finalOutputs)), hiddenOutputs.T)

        

        

    def test(self, inputs):

        a,b,c,d = self.forwardPropagation(inputs)

        return d
n = MultilayerPerceptron(2, 5, 2, 0.3)

for i in range(60):

    n.backPropagation(eingabewerte, zielwerte)

    i += 1

ergebnisse = []

for eingabe in evaleingabewerte:

    if n.test(eingabe)[0] > n.test(eingabe)[1]:

        ergebnis = -1

    else:

        ergebnis = 1

    ergebnisse.append(ergebnis)



index = 0

genauigkeit = 0

for ergebnis in ergebnisse:

    if ergebnis == evaldaten[index][-1]:

        genauigkeit += 1

    index += 1

print("Genauigkeit:", genauigkeit / len(ergebnisse) * 100, "%")