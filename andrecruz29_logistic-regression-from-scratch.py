import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
#In this section, N will be the size of our train set.
#I do not know the best value for this. 
#So I chose the one that got me best results for less Entropy. 

N=100
D=12
#Here I am trying to separate data from the targets.
#I am also slicing these arrays into train and test.
#Finally, I have arbitrarily decided that 7 or more 
#in quality is considered a 'Good Wine'.
 
X = np.asarray(df.drop('quality', axis=1))
X = (X-X.mean())/ np.std(X)
ones = np.ones((len(df['quality']),1))
Xb = np.concatenate((ones, X), axis=1)
Xb_t = Xb[N:,:] 
Xb = Xb[:N,:]

T = np.asarray(df['quality'])
for i in range(1599):
    if T[i] >= 7:
        T[i] = 1
    else:
        T[i] = 0

T_t = T[N:]
T = T[:N]
w = np.random.randn(D)/np.sqrt(D)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(Xb.dot(w))

def cross_entropy(T,Y):
    E=0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learn_rate=0.001
for i in range(10000):    
    w += learn_rate*(Xb.T.dot(T-Y))
    Y = sigmoid(Xb.dot(w))
    
def classification_rate(T,Y):
    return np.mean(T==Y)

print('Cross Entropy:', cross_entropy(T,Y))

Y=np.round(Y)

print('Classification Rate Train:',classification_rate(T,Y))

#Predictions

Yhat = sigmoid(Xb_t.dot(w))
Yhat = np.round(Yhat)

print('Classification Rate Test:',classification_rate(T_t,Yhat))

#This graphic shows the weights for each aspect of the wine. 
#I have noted that my Cross Entropy is still high.
#The predictions seem to be ok, though. 
#Raising N also increases the test accuracy up to 91%. 

labels = df.columns
plt.barh(labels[:-1], w[1:], color='g')
plt.show()