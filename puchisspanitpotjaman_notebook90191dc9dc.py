import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class Layer:
    def __init__(self, wide, length, activation='relu'):
        self.weight = np.random.normal(0,1,(wide,length))
        self.bias = np.zeros((length,))
        self.save = {}
        
    def forward(self, input):

        output = sigmoid(input.dot(self.weight) + self.bias)
        
        self.save['x'] = input
        self.save['y'] = output
        return output
    
    def backward(self, derror):
        
        dfunction_error = dsigmoid(self.save['y']) * derror
        dweight = self.save['x'].T.dot(dfunction_error)
        dbias = np.sum(dfunction_error)
        dx = dfunction_error.dot(self.weight.T)
        
        self.save['dweight'] = dweight
        self.save['dbias'] = dbias
        
        return dx

class NNModel:
    def __init__(self, h_layers, lr = 1):
        self.lr = lr
        self.layers = [Layer(a,b) for a,b in zip(h_layers[:-1], h_layers[1:])]
        
    def forward(self, input):
        res = input.copy()
        
        for layer in self.layers:
            res = layer.forward(res)
            
        return res
    
    def backward(self, dl):
        dl_h = dl.copy()
        for layer in self.layers[::-1]:
            dl_h = layer.backward(dl_h)
            
    def update(self):
        for layer in self.layers:
            layer.weight -= self.lr * layer.save['dweight']
            layer.bias -= self.lr * layer.save['dbias']
            
    def train(self, inp, actual, show = False):
        out = self.forward(inp)
        eps = 1e-5
        acc = np.mean(np.round(out) == actual)
        derror = - (actual / (out + eps) - (1 - actual) / (1 - out + eps)) / out.shape[0]
        
        if show:
            return out
        self.backward(derror)
        self.update()
        
        return acc
    
XOR_dataset = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

XOR_label = np.array([
    [0,1,1,0]
]).T


        
model = NNModel([2,2,1], lr = 10)

for i in range(1000):
    acc = model.train(XOR_dataset, XOR_label)
    if (i+1) % 50 == 0:
        print(f'epoch{i+1}: acc = {acc}')
print(model.train(XOR_dataset, XOR_label,True))

# model = NNModel([2,128, 128, 128,1], lr=1e-1)

# for i in range(10000):
    
#     X = np.array([np.random.randint(-3, 3, 20), np.random.randint(-3, 3, 20)]).T
    
#     y = (X.sum(axis=1) % 2).reshape((20,1))
    
#     acc = model.train_batch(X, y)
#     if (i+1) % 100 == 0:
#         print(f'epoch {i+1}: acc = {acc}')
# for i in range(-3,4):
#     for j in range(-3,4):
#         print(i)
#         print(j)
#         print(model.forward(np.array([i,j])))
