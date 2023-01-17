import numpy as np
def NeuNet(x1, x2, w1, w2, b):
    o = x1*w1 + x2*w2 + b
    return sigmoid(o)

def sigmoid(o):
    return 1/(1+np.exp(-o))
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()
NeuNet(1, 2, w1, w2, b)
