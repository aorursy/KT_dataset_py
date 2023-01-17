# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

from collections import Counter





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/pokemon/Pokemon.csv")
data.head(10)
data.info()
data.rename(columns={"Type 1":"Type1", "Type 2":"Type2", "Sp. Atk":"Sp.Atk", "Sp. Def":"Sp.Def"},inplace=True)

data.columns = data.columns.str.strip()
print("Types:",data["Type1"].unique().tolist())

print("Amount of types:",len(data["Type1"].unique().tolist()))
data.drop("Type2",axis = 1,inplace=True)
data["Type1"].value_counts()
# objects = ("Water","Normal","Grass","Bug","Psychic","Fire","Rock","Electric","Dragon","Ground","Ghost","Dark","Poison","Fighting","Steel","Ice","Fairy","Flying")

# y_pos = np.arange(len(objects))

# performance = [112,98,70,69,57,52,44,44,32,32,32,31,28,27,27,24,17,4]

# 

# plt.figure(figsize=(22,10))

# plt.bar(y_pos, performance, align='center', alpha=0.6)

# plt.xticks(y_pos, objects)

# plt.xlabel("Types")

# plt.ylabel('Frequency')

# plt.title('Pokemon Types')

# plt.show()
#CATEGORİCAL PLOTTING

var= data["Type1"]

#count number of categorical variable(value/sample)

varValue = var.value_counts()



#visualize

plt.figure(figsize=(20,10))

plt.bar(varValue.index,varValue)

plt.xticks(varValue.index,varValue.index.values) #With this command we can limit the ticks  

plt.ylabel("Frequency")

plt.title("Pokemon Types")

plt.show()
def plot_bar(variable):

    

    var = data[variable]

    varValue = var.value_counts()

    

    plt.figure(figsize=(15,10))

    plt.bar(varValue.index,varValue)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distrubution with barplot".format(variable))

    plt.show()
numericVar = ["HP","Attack","Total","Defense","Sp.Atk","Sp.Def","Speed"]

for n in numericVar:

    plot_bar(n)
data.drop(["Name"],axis=1,inplace=True)

data.drop(["#"],axis =1, inplace=True)
#List Comprehension 

data["Type1"] = [0 if i == "Water" else 1 if i == "Normal" else 2 if i== "Grass" else 3 if i=="Bug" else 4 if i=="Fire" or i=="Psychic" else 5 if i=="Electric" or i=="Rock"

                 else 6 if i=="Dragon" or i=="Ground" else 7 if i=="Ghost" or i=="Dark" else 8 if i=="Poison" or i=="Fighting" else 9 if  i=="Steel" or i=="Ice" else 10 for i in data["Type1"]]
#CATEGORİCAL PLOTTING

var= data["Type1"]

#count number of categorical variable(value/sample)

varValue = var.value_counts()



#visualize

plt.figure(figsize=(20,10))

plt.bar(varValue.index,varValue)

plt.xticks(varValue.index,varValue.index.values) #With this command we can limit the ticks  

plt.ylabel("Frequency")

plt.title("Pokemon Types")

plt.show()
data.info()
data.loc[:,"Type1"].value_counts()

y = data.Type1.values

x_data = data.iloc[:,2:]

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)) #Normalization



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
def initialize_weights_and_bias_NN(x_train,y_train):

    parameters = {"weight1":np.random.randn(3,x_train.shape[0])*0.1, #3 x 600

                  "bias1":np.zeros((3,1)),

                  "weight2":np.random.randn(y_train.shape[0],3)*0.1, #600 x 3

                  "bias2":np.zeros((y_train.shape[0],1))} 

        # We define 3x1 because z2 = w2*A1 + b2 then A1 is 3x1 and

        # when we want to matris product w2 must be 1x3 

    return parameters 



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
def forward_propagation_NN(x_train,parameters):

    Z1 = np.dot(parameters["weight1"],x_train) + parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)

    

    cache={"Z1":Z1,

           "A1":A1,

           "Z2":Z2,

           "A2":A2}

    return A2,cache
def loss_cost_NN(A2,y,parameters): #A2 is an output actually but we gonna use it input for cost function

    logaritmicprobs = np.multiply(np.log(A2),y)

    cost = -np.sum(logaritmicprobs)/y.shape[1]

    return cost
def backward_propagation_NN(cache,parameters,x,y): #Derivative Part

    dZ2 = cache["A2"]-y

    dW2 = np.dot(dZ2,cache["A1"].T)/x.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/x.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,x.T)/x.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/x.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
def update_parameters_NN(parameters,grads,learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                      "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                      "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                      "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
def prediction_NN(parameters,x_test): #Remember predictions always apply w/test data

    A2,cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5 our prediction is sign one (y_head=1),

    # if z is equal or smaller than 0.5, our prediction is sign zero (y_head=0) >> Sigmoid func

    

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

            

    return Y_prediction
def two_layer_NN(x_train,y_train,x_test,y_test,num_iterations):

    cost_list = []

    index_list = []

    #initializing parameters and layer size

    parameters = initialize_weights_and_bias_NN(x_train,y_train)

    

    for i in range(0,num_iterations):

        #Forward propagation 

        A2,cache = forward_propagation_NN(x_train,parameters)

        #Computing Cost 

        cost = loss_cost_NN(A2,y_train,parameters)

        #Backward Propagation

        grads = backward_propagation_NN(parameters,cache,x_train,y_train)

        #Updating Parameters 

        parameters = update_parameters_NN(parameters,grads)

        

        if i % 50 == 0: #Per 50 iterations

            cost_list.append(cost)

            index_list.append(i)

            print("Cost after Iteration %i: %f" %(i,cost))

            

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation = 45)

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    

    #Prediction 

    y_prediction_test = prediction_NN(parameters,x_test)

    y_prediction_train  = prediction_NN(parameters,x_train)

    

    #Printing Train & Test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_NN(x_train,y_train,x_test,y_test,num_iterations=2500)
#reshape - keras requires tranpose of data

x_train,x_test,y_train,y_test = x_train.T, x_test.T ,y_train.T, y_test.T
# L-layer NN w/Keras

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score #cross_val_score = 2 means 4 part train 1 part test

from keras.models import Sequential # Initializing NN Library

from keras.layers import Dense #Layer Builder



def build_classifier():

    classifier = Sequential() #İnitializing NN

    classifier.add(Dense(units = 16, kernel_initializer = "uniform", activation ="relu", input_dim = x_train.shape[1])) # First Hidden layer, we defined the input dimension 600

    classifier.add(Dense(units = 12, kernel_initializer = "uniform", activation ="relu"))

    classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation ="relu"))

    classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation ="relu"))

    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation ="relu"))

    classifier.add(Dense(units = 2, kernel_initializer = "uniform", activation ="relu"))

    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid")) #means 1 output (1 node) and if we want to create model after hidden layer to output activation = sigmoid

    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    return classifier



#lets call our classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)

accuracies = cross_val_score(estimator = classifier, X=x_train, y=y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy Mean :"+str(mean))

print("Accuracy Variance :"+str(variance))