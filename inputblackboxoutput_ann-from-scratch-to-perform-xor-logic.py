import numpy as np
import matplotlib.pyplot as plt
'''Activation function: Sigmoid function'''
sigmoid = lambda d:1/(1 + np.exp(-1*d))

'''Loss function: Half Mean Squared Error'''
compute_loss = lambda y, h:(1/(2*len(y)))*(np.sum((y-h)**2)) 

'''Intialize ANN weights and biases with random values'''
def random_init(nx,nh,ny):
    parameters = {}
    parameters['W1'] = np.random.rand(nh,nx)
    parameters['b1'] = np.random.rand(nh,1)
    parameters['W2'] = np.random.rand(ny,nh)
    parameters['b2'] = np.random.rand(ny,1)
    return parameters

'''Forward propagation in  ANN'''
def forward(in_data,parameters):
    w1 = parameters['W1']
    b1 = parameters['b1']
    w2 = parameters['W2']
    b2 = parameters['b2']
    
    zh1 = np.dot(w1,in_data) + b1
    h1 = sigmoid(zh1)
    
    zo = np.dot(w2,h1) + b2
    y_out = sigmoid(zo)
    
    return y_out,zo,h1,zh1
def model_X3H2Y1_MSE(x, y):
    # Intialize model parameters
    nx = x.shape[0]
    nh = 2
    ny = y.shape[0]

    parameters = random_init(nx,nh,ny)
    y_out,zo,h1,zh1 = forward(x,parameters)

    epoch = []
    error = []
    grads = {}
    for i in range(50):
        # Compute error in prediction
        loss = compute_loss(y,y_out)
        epoch.append(i)
        error.append(loss)
        
        # Update weights & biases
        grads['dyout'] = (1/ny)*(y_out - y)
        grads['dzo'] = grads['dyout']*y_out * (1 - y_out)
        grads['dW2'] = np.dot(grads['dzo'], h1.T)
        grads['db2'] = grads['dzo']
        grads['dh1'] = grads['dzo'] * parameters['W2'].T
        grads['dzh1'] = grads['dh1']*(1 - grads['dh1'])
        grads['dW1'] = np.dot(grads['dzh1'], x.T)
        grads['db1'] = grads['dzh1']

        parameters['W2'] -= grads['dW2']
        parameters['b2'] = parameters['b2'] - grads['db2']
        parameters['W1'] -= grads['dW1']
        parameters['b1'] = parameters['b1'] - grads['db1']

        # Predict output
        y_out,zo,h1,zh1 = forward(x, parameters)
        
    # Plot error v/s epoch figure
    plt.figure()
    plt.title('Plot of error v/s epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.plot(epoch, error)
    plt.show()

    # Print target and predicted values
    y_out = (y_out>0.5) * 1 
    print(f'Target:   {y[0]}')
    print(f'Predicted:{y_out[0]}')
inpt = np.array([[0,0,0,0,1,1,1,1],[0,0,1,1,0,0,1,1],[0,1,0,1,0,1,0,1]])
oupt = np.array([[0,1,1,0,1,0,0,1]])

model_X3H2Y1_MSE(inpt, oupt)
# EOF