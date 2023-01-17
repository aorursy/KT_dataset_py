import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/column_2C_weka.csv",sep = ",")

data.head()
data.info()
data.describe()
color_list = ['red' if i == "Abnormal" else 'green' for i in data.loc[: ,'class']]

pd.plotting.scatter_matrix(data.loc[:,data.columns != 'class'],

                          c = color_list,

                          figsize = [15,15],

                          diagonal = 'hist',

                          alpha = 0.5,

                          s = 200,

                          marker = '*',

                          edgecolor = "black")

plt.show()

sns.countplot(x = "class",data=data)

data.loc[:,'class'].value_counts()


x  = data.loc[:,data.columns != 'class']

y = data.loc[:,"class"]



# x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values





print(x.shape)

print(y.shape)



from sklearn.model_selection import  train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 , random_state = 1)





from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3)



knn.fit(x_train,y_train)



prediction = knn.predict(x_test)



print("{} knn score : {}".format(3,knn.score(x_test,y_test)))





print(len(x_train),len(y_train))
# model complexity

neig = np.arange(1,25)

train_accuracy = []

test_accuracy = []



for i ,k in enumerate(neig):

    knn=KNeighborsClassifier(n_neighbors=k)

    knn.fit(x_train,y_train)

    train_accuracy.append(knn.score(x_train,y_train))

    test_accuracy.append(knn.score(x_test,y_test))

    

#plot



plt.figure(figsize = [13,8])

plt.plot(neig,test_accuracy,label='testing accuracy')

plt.plot(neig,train_accuracy,label='train accuracy')

plt.legend()

plt.title('values vs accuracy')

plt.xlabel('number of neighbours')

plt.ylabel('accuracy')

plt.xticks(neig)

plt.show()

print('best accuracy is {} with k = {}'.format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))







data.loc[:,"class"] = [1 if each == "Abnormal" else 0 for each in data.loc[:,"class"]]

data.loc[:,"class"]
y = data.loc[:,"class"]

x_data = data.drop(["class"],axis = 1)



print(x_data.shape)

print(y.shape)

x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T.values.reshape(248,1)

y_test = y_test.T.values.reshape(62,1)



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)



#parameter initialize and sigmoid function

#dimension = 30



def initialize_weights_and_bias(dimension):

    w =np.full((dimension,1),0.01)

    #np.full içine aldığı (x,y),z ile x e y boyutlu z lerden oluşan bir arrray yapar

    b=0.0

    return w,b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z)) #aynı zamanda bu sigmoid fonksiyon 

    #np.exp e üzeri demek

    return y_head



def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    z = np.dot(w.T,x_train)+b #?

    #np.dot matrix çarpımında kullanılıyor

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1] #?

    

    #backward propagaiton

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias":derivative_bias}

    return cost, gradients



    

    

#update

def update (w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    #updating(learning) parameters is number_of_iteration time

    for i in range(number_of_iteration):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

    #lets upgrade

    

    w=w-learning_rate*gradients["derivative_weight"]

    b=b-learning_rate*gradients["derivative_bias"]

    

    if i%10 == 0:

        cost_list2.append(cost)

        index.append(i)

        print("cost after iteration %i:%f" %(i,cost))

        

    #we update(learn) parameters weight and bias



    parameters = {"weight":w,"bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("number of iteration")

    plt.ylabel("cost")

    plt.show()

    return parameters,gradients,cost_list

#prediction

def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            y_predction[0,i]=0

        else:

            y_prediction[0,i]=1

        return y_prediction

    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

   

    dimension =  x_train.shape[0]  

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    #print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    #print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 150)