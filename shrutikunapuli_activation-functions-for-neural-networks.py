import numpy as np

def sigmoid(z):

 return 1 / (1 + np.exp(-z))

print("Sigmoid of 4 is:",sigmoid (4))
#Non-zero centric
print("Sigmoid of positive number(5) is:",sigmoid(5))
print("Sigmoid of negative number(-5) is:",sigmoid(-5))
print("Difference between Derivative of Sigmoid (5) and (-5) is:", sigmoid(5)*(1- sigmoid(5))-sigmoid(-5)*(1- sigmoid(-5)))
#vanishing gradient
print("Difference between sigmoid of 14 and 15:",sigmoid(15)-sigmoid(14))
def tanh(z):

 return np.tanh(z)
print("tanh of 4 is:",tanh(4))
#zero-centric 
print("tanh of positive number(15) is:",tanh(15))
print("tanh of positive number(-15) is:",tanh(-15))
#vanishing gradient



print("Difference between tanh of 14 and 15:",np.tanh(15)-np.tanh(14))
def relu(z):

  return max(0, z)
z= 10

print("ReLU of "+str(z)+" is :",relu(z))
# Dead neuron

z= -0.4

print("ReLU of "+str(z)+" is :",relu(z)) 

z= -50

print("ReLU of "+str(z)+" is :",z * (z > 0))
def leakyrelu(z):

  return np.maximum(0.01 * z, z)
z= 10

print("ReLU of "+str(z)+" is :",leakyrelu(z)) #positive value
z= -1

print("ReLU of "+str(z)+" is :",leakyrelu(z)) #negative number
def parmetricrelu(z,α):

  return np.maximum(α * z, z)


print("ReLU of "+str(z)+" is :",parmetricrelu(10,0.5)) #positive value



print("ReLU of "+str(z)+" is :",parmetricrelu(-1,0.5)) #negative number

def erelu(z,alpha):

    return z if z >= 0 else alpha*(np.exp(z) -1)
print("Exponential ReLu for 10 is :",erelu(10,3))

print("Exponential ReLu for -10 is :",erelu(-10,3))
def softmax(x):

    ex = np.exp(x)

    sum_ex = np.sum( np.exp(x))

    return ex/sum_ex





print ("softmax of 1,2 and 3 is:",softmax([1,2,3]))