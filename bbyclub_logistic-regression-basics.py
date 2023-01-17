# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")

data.drop(["Unnamed: 32","id"],axis=1, inplace=True) #kullanmayacağım kolonları drop ediyorum. axis=1 diyerek tüm kolonu drop ettim .

#axis=0 deseydim tüm satır drop edilirdi. inplace=True ise drop sonrası datayı güncelle demek.

data.diagnosis= [1 if each=="M" else 0 for each in data.diagnosis] #string olan diagnosis etiketlerini 0 ve 1 olarak güncelledim 

# yani object veri türünü integer a çevirdim

print(data.info())



y=data.diagnosis.values # y benim datanım sınıfları

x_data=data.drop(["diagnosis"],axis=1) # y kolonunu çıkardıktan sonra geriye kalan tüm feature lar benim x imi yani inputları oluşturuyor
#%% normalization

#Tüm featureları normalize ediyoruz (yani 0 ile 1 arasında scale ediyoruz) ki 

#büyük değere sahip featurelar küçük olanları override etmesin yani galebe çalmasın :)

# np.min(x_data) --> Tüm kolonlara ait minimum değerleri verir

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) #Verinin %20 sini test

#%80 inin train olarak ayırdım. random_state=42 diyerek 42 gibi bir rastgele id verdik random işlemine ki

# her çalıştığında aynı şekilde bölsün yani 42 no lu random sonucunu versin. Random işlemini sabitledik yani



#row ları feature olarak ayarlamak için hepsinin transpozunu aldık. Konunun dökümanı ile(kaggle) uyumlu olması için 

x_test=x_test.T

x_train=x_train.T

y_train=y_train.T

y_test=y_test.T



print("x_test= ",x_test.shape)

print("x_train= ",x_train.shape)

print("y_train= ",y_train.shape)

print("y_test= ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w= np.full((dimension,1),0.01) #weightleri rastgele atadık. dimension=30 için 30,1 lik 0.01 den oluşan bir matris oluşturduk.

    b=0.0

    return w,b



#w,b=initialize_weights_and_bias(30)



#sigmoid function tanımlıyoruz.

def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head



#print(sigmoid(0))

    

def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    # Forward propagation steps:

    # find z = w.T*x+b

    # y_head = sigmoid(z)

    # loss(error) = loss(y,y_head)

    # cost = sum(loss)

    z=np.dot(w.T,x_train)+ b

    y_head=sigmoid(z)

    

    # -(1-y)log(1-y_head) - y log y_head

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1] #x_train.shape[1] is for scaling

    

    #backward propagation

    derivative_weight= (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients= {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias} 

    #ağırlık ve bias' ın türevini saklayan dictionary

    

    return cost, gradients



def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    #updating(learning parameters is number_of_iteration times)

    for i in range(number_of_iteration):

        #make forward and backward propagation and find cost and gradients

        cost, gradients= forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        #lets update

        w=w-learning_rate*gradients["derivative_weight"]

        b=b-learning_rate*gradients["derivative_bias"]

        if i%10==0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

            

    #we update(learn) parametrs weights and bias

    parameters={"weight":w, "bias": b }

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Number of iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters,gradients, cost_list
def predict(w,b,x_test):

    #x_test is a input for forward propagation

    z=sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction= np.zeros((1,x_test.shape[1]))

    #if z is bigger than 0.5, our prediction is sign one (y_head=1,

    #if z is smaller than 0.5, our prediction is sign zero (y_head=0)

    for i in range (z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

            

    return Y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    #,nitialize

    dimension= x_train.shape[0] # feature sayısı

    w,b= initialize_weights_and_bias(dimension)

    #do not change learning rate

    parameters,gradients,cost_list= update(w,b,x_train,y_train,learning_rate,num_iterations)

    #aldığı paramtreler

    #w,b : wight ve biası update edecek

    #x_train ve y_train i kullanarak bir model oluşturacak

    

    #labellarını bilmediğim x_test' lerin predict edilen labelları y_prediction

    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)



    #y_prediction_train= predict(parameters["weight"],parameters["bias"],x_train)

    #ileri seviye konu

    

    #Print train/test Errors

    #print("train accuracy: {} %".format(100-np.mean(np.abs(y_prediction_train-y_ttrain))*100))

    #ileri seviye konu

    

    print("test accuracy: {} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))

    #1 olan bir label ı 1 olarak tahmin edersem bu sonuç 100 çıkara yani accuracy 100 olur. Tersi durumunda  acc 0 olur.     

    #np.mean kullanmamızın sebebi eldeki değerin vektör olmasıdır.

    #format yazarak süslü parantezin olduğu yere değeri koyuyoruz çıktıyı alırken. C3 taki gibi



logistic_regression(x_train,y_train,x_test,y_test,learning_rate=3,num_iterations=300)



#↓%%skleran with Logistic regression

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))