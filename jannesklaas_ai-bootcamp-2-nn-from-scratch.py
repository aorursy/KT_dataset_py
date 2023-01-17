import numpy as np
#Seed the random function to ensure that we always get the same result
np.random.seed(1)
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
#set up w0
W1 = 2*np.random.random((3,1)) - 1

#define X
X = np.array([[0,1,0],
              [1,0,0],
              [1,1,1],
              [0,1,1]])

#define y
y = np.array([[0,1,1,0]])
#do the linear step
z1 = np.dot(X,W1)
#pass the linear step through the activation function
A1 = sigmoid(z1)
#see what we got
print(A1)
import numpy as np
#Seed the random function to ensure that we always get the same result
np.random.seed(1)
#Variable definition
#set up w0
W1 = 2*np.random.random((3,1)) - 1

#define X
X = np.array([[0,1,0],
              [1,0,0],
              [1,1,1],
              [0,1,1]])
#define y
y = np.array([[0],
              [1],
              [1],
              [0]])

#b may be 0
b1 = 0
m = X.shape[0]
A0 = X
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Log Loss function
def log_loss(y,y_hat):
    N = y.shape[0]
    l = -1/N * np.sum(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
    return l

def log_loss_derivative(y,y_hat):
    return (y_hat-y)
losses = []
for i in range(22):
    #do the linear step
    z1 = A0.dot(W1) + b1
    #pass the linear step through the activation function
    A1 = sigmoid(z1)
    
    #Calculate loss
    loss = log_loss(y=y,y_hat=A1)
  
    #we use this to keep track of losses over time
    losses.append(loss)

    #Calculate derivative of L(w) with respect to z0
    dz1 = log_loss_derivative(y=y,y_hat=A1)
    
    #Calculate derivative of L(w) with respect to w0
    dW1 = 1/m * np.dot(A0.T,dz1)
    
    #Calculate derivative of L(w) with respect to b0
    db1 = 1/m * np.sum(dz1,axis=0,keepdims=True)
    
    #Update w0 accoarding to gradient descent algorithm
    #To keep things easy we will ignore the learning rate alpha for now, it will be covered in the next chapter
    W1 -= dW1

    #Update b0 accoarding to gradient descent algorithm
    b1 -= db1
#Plot losses over time
#As you can see our algorithm does quite well and quickly reduces losses
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()