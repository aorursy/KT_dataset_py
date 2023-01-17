import numpy as np
def relu(X):

   return np.maximum(0,X)  
relu(3)
relu(0)
relu(-3)
array = np.arange(-10,11,1)

print(array)
relu(array)
import matplotlib.pyplot as plt

plt.plot(array, relu(array),'r')
x = [1, 2,5,8,10]

w = [0.3,-0.7,0.3,0.2,-0.1]

b = [1] *5 
def k(x1,x2):

    w1 = 0.5

    w2 = -0.3

    

    return np.array(x1)*np.array(w1).T + np.array(x2)*np.array(w2).T
l = np.arange(-10,10,0.2)
t = np.arange(-5,5,0.1)
plt.figure(figsize=(12,8))

plt.plot(t,k(t,l),'r',label='linear func')

plt.plot(t,relu(t),'o',color='g',label='relu -5:5')

plt.plot(l,relu(l),'*-',color = 'y',label = 'relu -10:10')

plt.legend()
np.array(x)
np.array(w).T
c = np.array(x)*np.array(w).T+b
c
plt.plot(c,'r')
plt.plot(relu(c),'g')
x2 = [2, 2, 3, 5,4]

w2 = [0.1, -0.3, 0.5, 0.3, -0.1]

b2 = [1]*5
np.array(x2)
np.array(w2).T
w0 = b+b2
w0
linear_output =np.array(x)*np.array(w).T + np.array(x2)*np.array(w2).T
linear_output
plt.plot([0,1,2,3,4], linear_output,'r')
plt.plot([0,1,2,3,4], relu(linear_output),'g')
linear_output
relu_output = relu(linear_output)
relu_output
plt.plot([0,1,2,3,4], linear_output,'r')

plt.plot([0,1,2,3,4], relu_output, 'o-',color='g')
print('*'*150)
t = np.arange(-5,5,0.1)
plt.figure(figsize=(12,8))

plt.plot(t,t,'r')

plt.plot(t,relu(t),'o',color='g')
plt.figure(figsize=(12,8))

plt.plot(t,t+3,'r')

plt.plot(t,relu(t+3),'o',color='g')
plt.figure(figsize=(12,8))

plt.plot(t,t-2,'r')

plt.plot(t,relu(t-2),'o',color='g')
from sklearn.neural_network import MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=500,solver='sgd',alpha=0.01,early_stopping=True).fit(training_inputs, training_outputs)
user_input_ = int(input("Karesi alinacak ifadeyi girin: "))
print("Girilen: ", user_input_)

print("Tahmin edilen deger: {} ".format(regr.predict(np.array([[user_input_]]))))
plt.figure(figsize=(10,8))

plt.plot(t,t**2,'o-',color = 'g')