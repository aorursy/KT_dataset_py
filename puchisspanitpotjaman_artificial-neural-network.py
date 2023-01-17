import numpy as np
import random

class matrix:
    def __init__(self, neuron, length):
      self.array = np.random.normal(0,1,(length,neuron))
      self.der = None
      self.length = length
      self.bias = np.zeros((length,1)).reshape(self.length,1)
    def multi(self, input):
      output = np.dot(self.array,input)
      output += self.bias
      output = 1/(1+ np.exp(-output))
      self.der = (output*(1 - output)).reshape(self.length,1)
      return output
    def change(self, data):
      self.array += data
    def change_bias(self, data):
      self.bias += data
    def multi_ReLU(self,input):
      output = np.dot(self.array,input)
      output += self.bias
      self.der = output.copy()
      for i in range(self.length):
        if output[i][0] <= 0:
          output[i][0] *= 0.001
          self.der[i][0] = 0.001
        else:
          self.der[i][0] = 1
      return output
class NeuralNetwork:
    def __init__(self, neuron, length, layer):
      self.array = [matrix(neuron,length)]+[matrix(length, length)]*(layer-1)
      self.length = length
      self.layer = layer
      self.neuron = neuron
      self.end = matrix(length,1)
      self.output = None
      self.predict = None
      self.input_layer = [None]*layer
      self.end_input = None
    def set_input(self, input):
      self.input_layer[0] = np.array(input).reshape(self.neuron,1)
    def set_output(self, output):
      self.output = output
    def train(self):
      a = self.array[0].multi_ReLU(self.input_layer[0])
      for i in range(1,self.layer):
        self.input_layer[i] = a
        a = self.array[i].multi_ReLU(a)
      self.end_input = a
      self.predict = self.end.multi(a)
    def backpropagate(self):
#       print(self.output - self.predict)
      error_der = 0.01*(self.output - self.predict)*self.end.der[0]
#       print(self.end.der[0])
      change = error_der*self.end_input.transpose()
      change_bias = np.array([error_der]).reshape(1,1)
      change_layer = [None]*self.layer
      change_bias_array = [None]*self.layer
      for i in range(self.layer-1,-1,-1):
        if i == self.layer-1:
#           print(self.end.array)
#           print(self.array[i].der.transpose())
          change_layer[i] = self.end.array*error_der*self.array[i].der.transpose()
          change_bias_array[i] = change_layer[i].transpose()
#           print(change_layer[i])
        else:
#           print(i)
#           print(change_layer[i+1].transpose())
#           print(self.array[i+1].array)
#           print(self.array[i].der)
          change_layer[i] = (self.array[i].der*np.dot(self.array[i+1].array.transpose(),change_layer[i+1].transpose())).transpose()
#           print(change_layer[i])
          change_bias_array[i] = change_layer[i].transpose()
      self.end.change(change)
      self.end.change_bias(change_bias)
      for k in range(self.layer): 
        self.array[k].change(change_layer[k].transpose()*self.input_layer[k].transpose())
#         print(k)
        self.array[k].change_bias(change_bias_array[k])
    def training(self):
      self.train()
      self.backpropagate()
    def run(self):
      self.train()
    def show(self):
      return self.predict
    def show_array(self):
      for i in range(self.layer):
          print(self.array[i].array)
    def loss_function(self):
      print((self.output - self.predict)*(self.output - self.predict))/2

num = 49
num_node = 2
num_hidden = 5
a = NeuralNetwork(num_node, num_hidden, 5)
inputmat = [[i,j] for j in range(-3,4) for i in range(-3,4)]
outputmat = [(inputmat[i][0] + inputmat[i][1])%2 for i in range(num)]
rd = 0
while rd <= 10000:
   rd += 1
   for i in range(num):
       a.set_input(inputmat[i])
       a.set_output(outputmat[i])
       a.training()
# a.set_input([0,0])
# a.run()
# print(a.show())
# a.set_input([0,1])
# a.run()
# print(a.show())
# a.set_input([1,0])
# a.run()
# print(a.show())
print(inputmat)
print(outputmat)
for i in range(num):
  a.set_input(inputmat[i])
  a.set_output(outputmat[i])
  a.run()
  print(a.show())



