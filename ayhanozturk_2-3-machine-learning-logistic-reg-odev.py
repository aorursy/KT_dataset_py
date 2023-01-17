import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

    # import graph objects as "go"

import plotly.graph_objs as go

import os

print(os.listdir("../input"))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
data.info()
data.describe()
data.head()
data.tail()
data.Age.value_counts(dropna=False)
data.isnull().sum()
train_data = data.copy()

train_data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)

train_data["Embarked"].fillna(data['Embarked'].value_counts().idxmax(), inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)
train_data.isnull().sum()
#ortalama koyduktan sonra değişimi inceleme



plt.figure(figsize=(15,8))

ax = data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

data["Age"].plot(kind='density', color='teal')

ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)

train_data["Age"].plot(kind='density', color='orange')

ax.legend(['Raw Age', 'Adjusted Age'])

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
train_data.head()
## Create categorical variable for traveling alone

train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 5, 8)

train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)
train_data.head()
#create categorical variables and drop some variables

training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])

training.head()
training.drop('Sex_female', axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop('Ticket', axis=1, inplace=True)



final_train = training

final_train.head()
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)
final_train.head()
test_data = test_df.copy()

test_data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)

test_data["Fare"].fillna(data["Fare"].median(skipna=True), inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)



test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)



test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)



testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])

testing.drop('Sex_female', axis=1, inplace=True)

testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)



final_test = testing



final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)



final_test.head()
cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

x_data = final_train[cols]

y = final_train['Survived']



cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

x_test_data = final_test[cols]

# y Survived tahmin deliecek final_test de Survived sutunu yok

#y_test = final_test['Survived']
# %% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

x_test_normalization  = (x_test_data - np.min(x_test_data))/(np.max(x_test_data)-np.min(x_test_data)).values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
# Önemli nokta su satırlarda özelikler sutunlarda ise kaç adet örnek varsa o olmalı

# yani 455*30 455 örnek 30 input (örnek*input) normalde biz bunu 30*455 yani input*örnek şekline çevirmeliyiz

x_train = x_train.T

x_test = x_test.T

y_train =np.array([y_train]).reshape(-1,1)

y_train = y_train.reshape(-1,1).T

y_test =np.array([y_test]).reshape(-1,1)

y_test = y_test.reshape(-1,1).T



print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
# %% parameter initialize and sigmoid function

# dimension = 30

def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





# w,b = initialize_weights_and_bias(30)
def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head

# print(sigmoid(0))
# %%

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

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

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    

    #Yeni Plot

    # Creating trace2

    trace1 = go.Scatter(

                        x = index,

                        y = cost_list2,

                        mode = "lines+markers",

                        name = "teaching",

                        marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                        text= index,)

    data_plot = [trace1]

    layout = dict(title = 'Logistic_Regression',

                  xaxis= dict(title= 'Index = Iterarion',ticklen= 5,zeroline= False)

                 )

    fig = dict(data = data_plot, layout = layout)

    iplot(fig)

    #Eski

    

    #plt.plot(index,cost_list2)

    #plt.xticks(index,rotation='vertical')

    #plt.xlabel("Number of Iterarion")

    #plt.ylabel("Cost")

    #plt.show()

    return parameters, gradients, cost_list
def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    

    for i in range(z.shape[1]):

        print("Prediction rate % =",z[0,i])

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



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

    

    w_calculated=parameters["weight"]

    b_calculated=parameters["bias"]

    

    return w_calculated,b_calculated
w_learned,b_learned=logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 30000)  
kacinci_kisi=5

y_prediction_test = predict(w_learned,b_learned,np.array([x_test.iloc[:,kacinci_kisi]]).reshape(-1,1))

# Not y_test[0,kacinci_hasta] burdaki sıfır y_test shape 1*örnek sayısı dolasıyla ilk satırı almak için 

# arraylerde sıfırdan başladığı için ilk satıra karşı sıfır yazarsın

print("tahmin_edilen: {0} gercekte_olan {1} ".format(int(y_prediction_test),y_test[0,kacinci_kisi]))
np.array([x_test.iloc[:,kacinci_kisi]]).reshape(-1,1).shape
x_test_normalization.T.shape
# gerçekten bilinmeyen import edilen test data ile çalışma

y_prediction_test = predict(w_learned,b_learned,x_test_normalization.T)
x_test_normalization.shape[0]
for each in range(x_test_normalization.shape[0]):

    print("prediction: {0}".format(y_prediction_test[0,each]))

    

y_prediction_test.shape[1]
#dosya kayıt

index_number=np.arange(0,y_prediction_test.shape[1],1)

index=index_number.reshape(-1,1)

print(y_prediction_test.shape[1])

print(index.shape)

data_record = pd.DataFrame(y_prediction_test.T,index,columns = ["Column1"])

data_record.to_csv("../input/data_recorded.csv",index = False)
np.ones([1,5])
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
y_test
kacinci_kisi=3

#lr.predict(np.array([x_test.iloc[:,1]]))

#Not #Not burda shape (30,) bunu düzeltmek lazım o sebeple reshape yapıcam

# x_test.iloc[:,1].shape = (30,)

#düzeltme np.array([x_test.iloc[:,1]]) = (30,1)

print("tahmin_edilen: {0} gercekte_olan {1} ".format(int(lr.predict(np.array([x_test.iloc[:,kacinci_kisi]]))),y_test[0,kacinci_kisi]))