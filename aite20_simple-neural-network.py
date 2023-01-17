import numpy as np 
import pandas as pd
from sklearn.datasets import load_breast_cancer
inputs, outputs = load_breast_cancer(return_X_y=True)
inputs = inputs.T
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
inputs = Scaler.fit_transform(inputs)
ReLU = lambda    x: np.maximum(0, x)
tanh = lambda    x: np.tanh(x)
sigmoid = lambda x: 1 /(1+np.exp(-x))
W, b, Z, A = {}, {}, {}, {"A0":inputs}
dW, db, dZ = {}, {}, {}
def makeLayerForward(input_, units=None, activation="sigmoid", layer=1, init="Yes"):
    if (init == "Yes"):
        if (layer==1):
            weights_shape = np.random.randn(units, inputs.shape[0]) 
        else:
            weights_shape = np.random.randn(units, W["W"+str(layer-1)].shape[0])
     
        W["W"+str(layer)] = weights_shape
        b["b"+str(layer)] = np.zeros((units, 1))
    
    elif (init == "No"):
        pass

    # Calculating the activations.
    Z["Z"+str(layer)] = np.dot(W["W"+str(layer)], input_) + b["b"+str(layer)] 
    
    if (activation == "sigmoid"): 
        A["A"+str(layer)] = sigmoid(Z["Z"+str(layer)])
        layer_output = A["A"+str(layer)]
        
    elif (activation == "tanh"): 
        A["A"+str(layer)] = tanh(Z["Z"+str(layer)])
        layer_output = A["A"+str(layer)]
        
    elif (activation == "relu"): 
        A["A"+str(layer)] = ReLU(Z["Z"+str(layer)])
        layer_output = A["A"+str(layer)]
   
    
    return layer_output

def makeLayerBackward(input_, units=None, activation="sigmoid", layer=1, type_="hidden"):
    # Activations backward.
    sigmoid_backward = lambda x: sigmoid(x) * (1-sigmoid(x))
    relu_backward =  lambda x: (x >= 0) + (x < 0)*0.01
    tanh_backward = lambda x: 1 - np.power(tanh(x), 2)
    
    activation_dervative = 0
    if (activation == "sigmoid"): activation_dervative = sigmoid_backward
    elif (activation == "relu"): activation_dervative = relu_backward
    elif (activation == "tanh"): activation_dervative = tanh_backward    
    
    if (type_ == "output"):
        dZ["dZ"+str(layer)] = input_ - outputs
    elif (type_ == "hidden") :    
        dZ["dZ"+str(layer)] = np.dot(W["W" + str(layer+1)].T, dZ["dZ" + str(layer+1)])
        dZ["dZ"+str(layer)] *= activation_dervative(Z["Z"+str(layer)])
    
    dW["dW"+str(layer)] = 1./input_.shape[1] * np.dot(dZ["dZ"+str(layer)], A["A"+str(layer-1)].T)
    db["db"+str(layer)] = 1./input_.shape[1] * np.sum(dZ["dZ"+str(layer)], axis=1, keepdims=True)
    
    return dW["dW"+str(layer)], db["db"+str(layer)]

def Update_weights(W, b, dW, db, lr=0.001):
    W = W - lr*dW
    b = b - lr*db
    
    return W, b
# Forward propagation step.
layer_1_output = makeLayerForward(input_=inputs, units=30,
                                  activation="tanh", layer=1, init="Yes")

layer_2_output = makeLayerForward(input_=layer_1_output, units=20,
                                  activation="relu", layer=2, init="Yes")

layer_3_output = makeLayerForward(input_=layer_2_output, units=10,
                                  activation="relu", layer=3, init="Yes")

layer_4_output = makeLayerForward(input_=layer_3_output, units=1,
                                  activation="sigmoid", layer=4, init="Yes")
for i in range(5000):
    # Forward propagation step.
    layer_1_output = makeLayerForward(input_=inputs, units=30,
                                      activation="tanh", layer=1, init="No")
    
    layer_2_output = makeLayerForward(input_=layer_1_output, units=20,
                                      activation="relu", layer=2, init="No")
    
    layer_3_output = makeLayerForward(input_=layer_2_output, units=10,
                                      activation="relu", layer=3, init="No")
    
    layer_4_output = makeLayerForward(input_=layer_3_output, units=1,
                                      activation="sigmoid", layer=4, init="No")
    
    
    # Backward propagation step.
    dW4, db4 = makeLayerBackward(input_=layer_4_output, units=1,
                                 activation="sigmoid", layer=4, type_="output")
    
    dW3, db3 = makeLayerBackward(input_=layer_3_output, units=10,
                                 activation="relu", layer=3)
    
    dW2, db2 = makeLayerBackward(input_=layer_2_output, units=20,
                                 activation="relu", layer=2)
    
    dW1, db1 = makeLayerBackward(input_=layer_1_output, units=30,
                                 activation="tanh", layer=1)

    # Updating weights.
    W["W1"], b["b1"] = Update_weights(W["W1"], b["b1"], dW1, db1, lr=0.02)
    W["W2"], b["b2"] = Update_weights(W["W2"], b["b2"], dW2, db2, lr=0.02)
    W["W3"], b["b3"] = Update_weights(W["W3"], b["b3"], dW3, db3, lr=0.02)
    W["W4"], b["b4"] = Update_weights(W["W4"], b["b4"], dW4, db4, lr=0.02)
    
    # Displaying the accuracy score.
    print("Acc: ", np.mean((layer_4_output>=0.5) == outputs)*100)
print((layer_4_output>=0.5).astype(int)[0][:30])
print(outputs[:30])
