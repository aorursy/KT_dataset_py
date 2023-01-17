import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
train_y=np.array((train[["label"]]).T)
train_x=np.array(train)
train_x=np.delete(train_x,0,1).T
test_x=np.array(test).T
train_x=train_x/255  #Normalizing the input data
test_x=test_x/255
digits = 10
examples = train_y.shape[1]
Y_new = np.eye(digits)[train_y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)
i = 40
plt.imshow(train_x[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
Y_new[:,i]
def compute_cost(Y,Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    
    return L
def sigmoid(Z):
    return 1/(1+np.exp(-Z))
n_x=train_x.shape[0]
n_h=64                #number of nodes in hidden layer
learning_rate=0.75
W1 = np.random.randn(n_h, n_x)
b1 = np.zeros((n_h, 1))
W2 = np.random.randn(digits, n_h)
b2 = np.zeros((digits, 1))
m=train_x.shape[1]
X=train_x
Y=Y_new

for i in range(3000):  
    #Implementation of forward propagation
    Z1=np.dot(W1,X) +b1     
    A1=sigmoid(Z1)
    Z2=np.dot(W2,A1)+b2
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
    
    cost=compute_cost(Y,A2)
    
    #Implementation of backward propagation
    dZ2=A2-Y              
    dW2=(np.dot(dZ2,A1.T))/m
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1=np.dot(W2.T,dZ2)
    dZ1 = dA1 * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    #Implementation of Gradient Descent
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    
    if (i % 100 == 0):
        print("Epoch", i, "cost: ", cost)
    
print("Final cost:", cost)
Z1 = np.matmul(W1, test_x) + b1
A1 = sigmoid(Z1)
Z2 = np.matmul(W2, A1) + b2
A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0)
predictions = np.argmax(A2, axis=0)    #Selecting the index with the max value which'll correspond to the prediction
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)