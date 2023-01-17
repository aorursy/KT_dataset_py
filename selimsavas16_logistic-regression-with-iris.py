import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff 
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/iris/Iris.csv")
data.head()
data.describe().T
data.info()
data.corr()
data.Species.unique()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.violinplot(data=data, x='Species',y='PetalLengthCm')
plt.subplot(2,2,2)
sns.violinplot(data=data, x='Species',y='PetalWidthCm')
plt.subplot(2,2,3)
sns.violinplot(data=data, x='Species',y='SepalLengthCm')
plt.subplot(2,2,4)
sns.violinplot(data=data, x='Species',y='SepalWidthCm')

dataSetosa = data[data.Species == 'Iris-setosa'] 
dataVersicolor = data[data.Species == 'Iris-versicolor']
dataVirginica = data[data.Species == 'Iris-virginica']

trace1 = go.Box(
    y=dataSetosa.SepalLengthCm,
    name = 'Setosa',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace2 = go.Box(
    y=dataVersicolor.SepalLengthCm,
    name = 'Versicolor',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)

trace3 = go.Box(
    y=dataVirginica.SepalLengthCm,
    name = 'Virginica',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

data1 = [trace1,trace2, trace3]
iplot(data1)
plt.figure(figsize = (25,10))

plt.subplot(2,2,1)
sns.swarmplot(x="SepalWidthCm", y="SepalLengthCm", hue="Species",data = data, palette="Paired")
plt.subplot(2,2,2)
sns.swarmplot(x="PetalWidthCm", y="PetalLengthCm", hue="Species",data = data, palette="Paired")

data.head()
dataSetosa = data[data.Species == 'Iris-setosa'] 
dataVersicolor = data[data.Species == 'Iris-versicolor']
dataVirginica = data[data.Species == 'Iris-virginica']


new_data = pd.concat([dataVersicolor,dataVirginica])
new_data = new_data.reset_index() 
new_data
# Kullanmadığımız columnsları drop edelim
# axis = 1 : tüm bir columns
# inplace : Drop et ve datanın içine kaydet
new_data.drop(["index","Id"], axis=1,inplace = True)
new_data.head()
# String barındıran columns değerimizi değiştirdik
new_data.Species = [1 if each == "Iris-versicolor" else 0 for each in new_data.Species]
new_data.head()
y = new_data.Species.values
x_data = new_data.drop(["Species"], axis=1) # Species değerlerimizin olmadığı bir veri seti

x_data.head()
# (x - x(min)/(max(x)- min(x))

# All columns must be in the range 0 to 1. Features should not differ greatly between them.

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state= 16)

# Transpoz process
x_train = x_train.T
x_test = x_test.T#
y_train = y_train.T
y_test = y_test.T


# shape values
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01) # Create values with 0.01 values
    
    b = 0.0 
    
    # These numbers are usually selected when using w and b.
    
    return w,b

# sigmoid func = 1/ (1+ e^(-x))

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    
    return y_head
    
print(x_train.shape[1])
def forward_backward_propagation(w,b,x_train,y_train):
    
    # forward propagation  
    
    # ((w values (1,30) * (4, 80) process) ve add bias values) = z
    z = np.dot(w.T,x_train) + b       # z values / np.dot : satır ve sütün olarak çarp
    y_head = sigmoid(z)               # The y_head value of our z value in s function
    
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) # loss value
    cost = (np.sum(loss))/x_train.shape[1]                      # /x_train.shape[1]=80   to normalized
    
    
    # backward propagation
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]              # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                                # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias} # we used dictionary to store parameters
    
    return cost,gradients
    
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 50 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
tahmini_deger = []
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
            tahmini_deger.append(0)
        else:  
            Y_prediction[0,i] = 1
            tahmini_deger.append(1)

    return Y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 1001)    


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
tahmin = np.array(tahmini_deger)
tahmin
y_test
Tahmin = np.concatenate((y_test,tahmin), axis=0)

Tahmin = Tahmin.reshape(2,20)

Tahmin
x_test.T # Data in index 33 failed under LR.
deger = x_test.T
deger = deger.reset_index() 
deger["Real_Values"] = ["Iris-versicolor" if each == 0 else "Iris-virginica" for each in y_test]
deger["Test_Values"] = ["Iris-versicolor" if each == 0 else "Iris-virginica" for each in tahmin]
deger["Success"] = -(y_test - tahmin)+1

deger
