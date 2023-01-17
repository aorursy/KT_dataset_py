from IPython.display import FileLinks

import numpy as np # linear algebra

import pandas as pd



import os

print(os.listdir("../input"))



import re

import pprint



import csv
test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')
test_data.head()
train_data.head()
for i, name in enumerate(train_data['Name']):

    if '"' in name:

        train_data['Name'][i] = re.sub('"', '', name)



for i, name in enumerate(test_data['Name']):

    if '"' in name:

        test_data['Name'][i] = re.sub('"', '', name)
with open('../input/train.csv', 'r') as f:

    reader = csv.reader(f, delimiter=',')

    next(reader)

    a = []

    strings = []

    b = []

    for row in reader:

        passengerId = row[0]

        survived = row[1]

        name = row[3]

        pclass = row[2]

        sex = row[4]

        age = row[5]

        sibsp = row[6]

        parch = row[7]

        ticket = row[8]

        fare = float(row[9])

        cabin = row[10]

        embarked = row[11]

        gender = 0

        if row[4] == 'male':

            gender = 1

        else:

            gender = 0

        junk = list([int(pclass), int(sibsp), int(parch), int(fare), int(gender)])

        stuff = list([name, sex, ticket, cabin, embarked])

        strings.append(stuff)

        if age != '':

            if age.isdigit():

                junk.append(int(age))

            elif age.isdecimal():

                junk.append(int(float(age)))

        else:

            junk.append(0)



        a.append(junk)

        b.append(int(survived))



    pprint.pprint(a)

    pprint.pprint(b)
print(len(a))

print(len(b))
def sigmoid(x):

  return 1/(1+np.exp(-x))



def sigmoid_derivative(x):

  return x * (1-x) 



training_inputs = np.array(a)



training_outputs = np.array(b).T



np.random.seed(1)



synaptic_weights = 2 * np.random.random((891,1)) - 1



print('Random starting synaptic weights: ')

print(synaptic_weights)



for iteration in range(891):

  

  input_layer = training_inputs

  

  outputs = sigmoid(np.dot(input_layer, synaptic_weights))

  

  error = training_outputs - outputs

  

  adjustments = error * sigmoid_derivative(outputs)

  

  synaptic_weights += np.dot(input_layer.T, adjustments)

    

  

print('Synaptic weights after training: ')

print(synaptic_weights)

  

print('Outputs after training: ')

print(outputs)
type(input_layer)
with open('../input/test.csv', 'r') as f:

    reader = csv.reader(f, delimiter=',')

    next(reader)

    c = []

    e = []

    for row in reader:

        passengerId = row[0]

        name = row[2]

        pclass = row[1]

        sex = row[3]

        age = row[4]

        sibsp = row[5]

        parch = row[6]

        ticket = row[7]

        fare = row[8]

        cabin = row[9]

        embarked = row[10]

        junk = list([int(passengerId), int(pclass), int(sibsp), int(parch), 7.0])

        stuff = list([name, sex, ticket, cabin, embarked])

        c.append(junk)

        e.append(stuff)

        """if age != '':

            if age.isdigit():

                inputs.append([int(age)])

            elif age.isdecimal():

                inputs.append([float(age)])"""



    pprint.pprint(c)

    pprint.pprint(e)
class NeuralNetwork():

  

  def __init__(self):

    np.random.seed(1)

    

    self.synaptic_weights = 2 * np.random.random((5, 1)) - 1

    

  def sigmoid(self, x):

    return 1/(1+ np.exp(-x))

  

  def sigmoid_derivative(self, x):

    return x * (1-x)

  

  def train(self, training_inputs, training_outputs, training_iterations):

    

    for iteration in range(training_iterations):

      

      output = self.think(training_inputs)

      error = training_outputs - outputs

      adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

      self.synaptic_weights += adjustments

      

  def think(self, inputs):

    

    inputs = inputs.astype(float)

    output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

    

    return output

  

if __name__ =="__main__":

  

  neural_network = NeuralNetwork()

  

  print('Random synaptic weights: ')

  print(neural_network.synaptic_weights)

  

  training_inputs = np.array(a)

  training_outputs = np.array([b]).T

  

  neural_network.train(training_inputs, training_outputs, 100000)

  

  print("Synaptic weights after training: ")

  print(neural_network.synaptic_weights)

  

out = []

for i in c:

    A = str(i[0])

    B = str(i[1])

    C = str(i[2])

    D = str(i[3])

    E = str(i[4])



    print("New situation: input data = ", A, B, C, D, E)

    print("Output Data: ")

    output_data = int(neural_network.think(np.array([A, B, C, D, E])))

    out.append(output_data)

    print(output_data)
with open('submission.csv', mode='w') as submission:

    sub_writer = csv.writer(submission, delimiter=',', quotechar='"', quoting = csv.QUOTE_MINIMAL)

    sub_writer.writerow(['PassengerID', 'Survived'])

    inc = 0

    for i in c:

        sub_writer.writerow([i[0], out[inc]])

        inc += 1



sub = pd.read_csv('submission.csv')

sub.head()
FileLinks('.')