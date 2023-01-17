import numpy as np # lineer cebir

import pandas as pd # veri işleme

import matplotlib.pyplot as plt #görselleştirme

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/framingham-heart-study-dataset/framingham.csv')
data.info()
data.drop(['education','cigsPerDay','BPMeds','totChol','heartRate','glucose','BMI'],axis=1,inplace=True)

data.head(10) # ilk 10 örneği göster
y = data.TenYearCHD.values

x_data = data.drop(['TenYearCHD'], axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values # values: numppy array'e çevirmek için

x.head()
print(data.info())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# satır ve sütunların yerlerini değiştiriyorum

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train: ",x_train.shape) #14 tane feature, 3392 tane sample var

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
def initialize_weights_and_bias(dimension): # dimension: kaç tane feature varsa o kadar weights oluştur

    w = np.full((dimension,1),0.01) # weigths'in tüm değerlerine 0.01 verdik

    b = 0.0 

    return w,b
def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head
print(sigmoid(0), sigmoid(10))
def forward_back_propagation(w, b, x_train, y_train):

    # forward propagation

    z = np.dot(w.T, x_train) + b # matris çarpımı için w'in transpozu alındı

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #hata fonksiyonu formülü

    cost = (np.sum(loss)) / x_train.shape[1] #hata fonksiyonlarını topla, x_train.shape[1] ile scale et

    

    # back propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1] #weight değerlerinin türevini al

    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1] #bias değerlerinin türevini al

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias} #parametreleri depolamak için (dictionary)

    

    return cost, gradients #gradients: weight'in cost'a göre türevi
#parametrelerin güncellenmesi

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    

    #number_of_iteration kadar parametreleri güncelle

    for i in range(number_of_iteration):

        #forward ve back propagation ile cost ve gradients'leri bulalım

        cost, gradients = forward_back_propagation(w, b, x_train, y_train)

        cost_list.append(cost) #tüm cost'ları depola

        #güncelleme

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0: 

            cost_list2.append(cost) 

            index.append(i)

            print("Güncellemeden sonra cost %i: %f" %(i, cost))

            

    #weights ve bias'ı parameters içerisinde depola

    parameters = {"weight": w,"bias": b}

        

    #görselleştirme

    plt.plot(index, cost_list2)

    plt.xticks(index, rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
#tahmin

def predict(w, b, x_test):

    #Forward Propagation için x_test giriş değerimiz

    z = sigmoid(np.dot(w.T, x_test)+b)

    Y_prediction = np.zeros((1, x_test.shape[1]))

    

    #z 0.5'ten büyükse tahmin sonucu 1 (y_head = 1)

    #z 0.5'ten küçükse tahmin sonucu 0 (y_head = 0)

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction

    
#logistic regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):

    #değişkenleri oluştur

    dimension = x_train.shape[0] #3392 tane

    w,b = initialize_weights_and_bias(dimension)

    

    

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    

    #train/test errors göster

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))



logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations = 200)
#Sklearn ile LR

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



lr.fit(x_train.T, y_train.T) #sample & feature

print("test accuracy: {}".format(lr.score(x_test.T, y_test.T)))