import numpy as np
import random

class matrix:
    def __init__(self, neuron, length):
      self.array = np.random.random((length,neuron))
      self.der = None
      self.length = length
    def multi(self, input):
      output = np.dot(self.array,input)
      for i in range(self.length):
        output[i] = 1/(1+ np.exp(-output[i]))
      self.der = np.array([ output[i]*(1 - output[i]) for i in range(self.length) ]).reshape(self.length,1)
      return output
    def change(self, data):
      self.array += data
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
      a = self.array[0].multi(self.input_layer[0])
      for i in range(1,self.layer):
        self.input_layer[i] = a
        a = self.array[i].multi(a)
      self.end_input = a
      self.predict = self.end.multi(a)
    def backpropagate(self):
#       print(self.output - self.predict)
      error_der = (self.output - self.predict)*self.end.der[0]
#       print(self.end.der[0])
      change1 = error_der*self.end_input.transpose()
      change_layer = [None]*self.layer
      for i in range(self.layer-1,-1,-1):
        if i == self.layer-1:
#           print(change1)
          change_layer[i] = error_der*self.end.array*self.array[i].der.transpose()
#           print(change_layer[i])
        else:
#           print(i)
#           print(change_layer[i+1])
          change_layer[i] = change_layer[i+1]*self.array[i].der.transpose()
      self.end.change(change1)
      for k in range(self.layer): 
        self.array[k].change(change_layer[k].transpose()*self.input_layer[k].transpose())
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
num = 4
num_node = 2
num_hidden = 4
a = NeuralNetwork(num_node, num_hidden, 2)
inputmat = [[0,0],[0,1],[1,0],[1,1]]
outputmat = [0,1,1,0]
count = 0
# for x in range(-3,4,+1):
#   for y in range (-3,4,+1):
#     inputmat[count] = [x,y]
#     if (x+y)%2 == 0:
#       outputmat[count] = 0
#     else:
#       outputmat[count] = 1
#     count += 1
rd= 1
while rd <= 200000:
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
for i in range(num):
  print("x,y =",inputmat[i])
  print("odd = 1 even = 0",outputmat[i])
  a.set_input(inputmat[i])
  a.run()
  print(a.show())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class Layer:
    def __init__(self, h1, h2, activation='relu'):
        self.h1 = h1
        self.h2 = h2
        self.w = np.random.normal(0,1,(h1,h2))
#         self.w = np.random.random((h1,h2))
        self.b = np.zeros((h2,))
        
        self.cache = {}
        
    def forward(self, inp):
        '''
        inp: N x h1
        
        Return: N x h2
        '''
        out = sigmoid(inp.dot(self.w) + self.b)
        
        self.cache['x'] = inp
        self.cache['h'] = out
        return out
    
    def backward(self, dl):
        '''
        dl: N x h2
        '''
        
        df = dsigmoid(self.cache['h']) * dl
        dw = self.cache['x'].T.dot(df)
        db = np.sum(df, axis=0)
        dx = df.dot(self.w.T)
        
        self.cache['dw'] = dw
        self.cache['db'] = db
        
        return dx
        
one_layer = Layer(2, 3)
        
        
eps = 1e-5

class NNModel:
    def __init__(self, h_layers, lr=1e-2):
        assert h_layers[-1] == 1
        
        self.layers = [Layer(h1, h2) for h1, h2 in zip(h_layers[:-1], h_layers[1:])]
        self.lr = lr
        
    def forward(self, inp):
        h = inp.copy()
        
        for layer in self.layers:
            h = layer.forward(h)
            
        return h
    
    def backward(self, dl):
        dl_h = dl.copy()
        
        for layer in self.layers[::-1]:
            dl_h = layer.backward(dl_h)
            
    def update(self):
        for layer in self.layers:
            layer.w -= self.lr * layer.cache['dw']
            layer.b -= self.lr * layer.cache['db']
            
    def train_batch(self, inp, label):
        out = self.forward(inp)
        
        acc = np.mean(np.round(out) == label)
        loss = - np.mean((1 - label) * np.log(1 - out + eps) + label * np.log(out + eps))
        dl = - (label / (out + eps) - (1 - label) / (1 - out + eps)) / out.shape[0]
        
        
        self.backward(dl)
        self.update()
        
        return acc, loss
        
        
XOR_dataset = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

XOR_label = np.array([
    [0,1,1,0]
]).T

model = NNModel([2,2,1], lr=1)

for i in range(1000):
    acc, loss = model.train_batch(XOR_dataset, XOR_label)
    if (i+1) % 50 == 0:
        print(f'epoch {i+1}: acc = {acc}, loss = {loss}')
# Test Output
model.forward(XOR_dataset)
# Generate sample

X = np.array([np.random.randint(-3, 3, 20), np.random.randint(-3, 3, 20)]).T
X, X.sum(axis=1) % 2
model = NNModel([2,128, 128, 128,1], lr=1e-1)

for i in range(10000):
    
    X = np.array([np.random.randint(-3, 3, 20), np.random.randint(-3, 3, 20)]).T
    
    y = (X.sum(axis=1) % 2).reshape((20,1))
    
    acc, loss = model.train_batch(X, y)
    if (i+1) % 100 == 0:
        print(f'epoch {i+1}: acc = {acc}, loss = {loss}')
model.forward(np.array([[1,-2]]))
