# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import warnings

import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
### load dataset

x_l= np.load('../input/X.npy')

Y_l= np.load('../input/Y.npy')



img_size=64

plt.subplot(1,2,1)

plt.imshow(x_l[260].reshape(img_size,img_size)) # x_l içinden 260. resmi göster

plt.axis('off') # x ve y eksenlerni gösterme sadece resmi göster

plt.subplot(1,2,2)

plt.imshow(x_l[900].reshape(img_size,img_size))

plt.axis('off')
#Join a sequence of arrays along a row axis

X=np.concatenate((x_l[204:409],x_l[822:1027]),axis=0) #from 0 to 204 is zero sign and from 205 to 410 is one sign

#64*64 lük bir resimden 410 tane

#x_l resimlerden oluşan bir dizi. Bu dizi içinden 205 er rasimlik iki ayrı resim dizisini (0 ve 1 resimleri) alıp tek dizide birleştiriyorum

#Tek satırlık 410 resimden oluşan bir resim vektörü elde etmiş oldum ()

(410, 64, 64)
z=np.zeros(205) #205 lik bir 0 dizisi. Henüz (205,1) boyutunda değil. reshape etmek gerekir.

o=np.ones(205) #205 lik bir 1 dizisi. Henüz (205,1) boyutunda değil. reshape etmek gerekir.

Y=np.concatenate((z,o),axis=0) #z ve o dizilerini x ekseni boyunca birleştir. (410,) boyutunda bir dizi elde edilir.

print(Y.shape) #deneme. 

Y=Y.reshape(X.shape[0],1) #0 lardan 205 tane 1' lerden 205 tane olmak üzere 410 tane class ((410,1)'lik vektör)

print("X shape: ", X.shape)

print("Y shape: ", Y.shape)
#Then lets create x_train, y_train,x_test,y_test arrays

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state=42)

#Modelin %15 i test gerisi train, random state ile random yapma işlemini aynı değerlere sabitledik

number_of_train= X_train.shape[0] #

number_of_test= X_test.shape[0]

X_train_flatten=X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2]) #Resimleri tek boyutlu bir dizi haline getirdik

X_test_flatten= X_test.reshape(number_of_test,X_test.shape[1]*X_test.shape[2]) #Resimleri tek boyutlu bir dizi haline getirdik.

#X ve y korrdinatlarını tek vektör haline getirdik. 64*64 lük matris 4096 lık vektör oldu.

print("X train flatten",X_train_flatten.shape)

print("X test flatten",X_test_flatten.shape)
x_train = X_train_flatten.T

x_test = X_test_flatten.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
#initialize parameters

#so what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b
# calculation of z

#z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
y_head = sigmoid(0)

y_head
#forward propagaation steps:

#find z= w.T*x+b

#y_head=sigmoid(z)

#loss(error)= loss(y,y_head)

#cost=sum(loss)



#In backward propagation we will use y_head that found in forward propagation

#Therefore instead of writing backward propagation, lets combine both forward and backward

def forward_backward_propagation(w,b,x_train,y_train):

    z= np.dot(w.T,x_train)+b

    y_head= sigmoid(z) #probabilistic 0-1

    loss= -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost= (np.sum(loss))/x_train.shape[1] #x_train.shape[1] is for scaling

    #backward propagation

    derivative_weight= (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #x_train.shape[1] is for scaling

    derivative_bias= np.sum(y_head-y_train)/x_train.shape[1] #x_train.shape[1] is for scaling

    gradients= {"derivative_weight":derivative_weight, "derivative_bias":derivative_bias}

        

    return cost,gradients

    
#Updating parameters

def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    #updating(learning) parameters is number_of_iteration

    for i in range(number_of_iteration):

        #make forward and backward propagation and find cost and gradients

        cost,gradients= forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        #lets update

        w=w-learning_rate*gradients["derivative_weight"]

        b=b-learning_rate*gradients["derivative_bias"]

        if i%10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

        #we update(learn) parameters weight and bias

    parameters= {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters,gradients,cost_list

    #parameters, gradients, cost_list = update(w,b,x_train, y_train, learning rate= 0.009,number_of_iteration=200)     
#prediction

def predict(w,b,x_test):

    #x_test is a input for forward propagation

    z=sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction=np.zeros((1,x_test.shape[1]))

    #if z is bigger than 0.5 our prediction is sign one (y_head=1)

    #if z is smaller than 0.5 our prediction is sign zero(y_head=0)

    for i in range (z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

             Y_prediction[0,i]=1

    return Y_prediction

#predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    #initialize

    dimension=x_train.shape[0] #4096

    print(dimension)

    w,b = initialize_weights_and_bias(dimension)

    #do not change learning rate

    parameters, gradients,cost_list= update(w,b,x_train,y_train,learning_rate,num_iterations)

    

    y_prediction_test= predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train= predict(parameters["weight"],parameters["bias"],x_train)

    

    #Print train/test Errors

    print("train accuracy: {} %".format(100-np.mean(np.abs(y_prediction_train- y_train))*100))

    print("test accuracy: {} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))
logistic_regression(x_train,y_train,x_test, y_test, learning_rate=0.01, num_iterations=150)