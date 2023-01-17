# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# load data set

x_l = np.load('../input/Sign-language-digits-dataset/X.npy')#npy uzantılı data seti okuduk.

Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')

img_size = 64# resmin formatını 64 olarak ayarlayacaz çünkü data set açıklamalarında bize 64 şeklinde oalcagını belirtti.

plt.subplot(1, 2, 1)

plt.imshow(x_l[260].reshape(img_size, img_size))#reshape biçimlendirme diye geçer matplotlib de 64 64 yaprak resmi gösterdi

plt.axis('off')#buradaki off silersen kenarlıklardaki değerler gelmez bu off dursun aşagıdaki off siliyorum ikisini karşılaştırırsın

plt.subplot(1, 2, 2)

plt.imshow(x_l[900].reshape(img_size, img_size))

plt.axis()

plt.show()
# Join a sequence of arrays along an row axis.

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # 409-204=205  1027-822=205 toplam=410 axis=0 yukardan aşagı birleştir demek

z = np.zeros(205)

o = np.ones(205)

Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)

print("X shape: " , X.shape)

print("Y shape: " , Y.shape)
# Then lets create x_train, y_train, x_test, y_test arrays

#Normalde 410 resim var bu resimler %15 test olarak kullanılacak yani 62 tanesi geriye kalan 348 tanesi ise traing için yani eğitmek için kullanılacak

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]

print(number_of_train)

print(number_of_test)
print(number_of_train)
X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])

X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

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
# short description and example of definition (def)

def dummy(parameter):

    dummy_parameter = parameter + 5

    return dummy_parameter

result = dummy(3)     # result = 8



# lets initialize parameters

# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):#buradaki dimension bir resmin boyutu

    w = np.full((dimension,1),0.01)#4096 çarpı 1 lik bir matris oluşturur ve içerisine 0.01 değerini atar np.full 4096,1 lik kadar matris oluşturur.

    b = 0.0#bias değeri

    return w, b 

w,b=initialize_weights_and_bias(4096)

print(initialize_weights_and_bias(4096))

#w.shape w nün içerisinde nasıl bir matris oluştugunu görürsün
w.shape
#w,b = initialize_weights_and_bias(4096)
# calculation of z

#z = np.dot(w.T,x_train)+b# w.T buna takılma dedi matris işlemleri için yaptık

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head # y_head değeri elde ettik 0 ile 1 arasında
y_head = sigmoid(10)

y_head
# Forward propagation steps:

# find z = w.T*x+b

# y_head = sigmoid(z)

# loss(error) = loss(y,y_head)

# cost = sum(loss)

def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) # probabilistic 0-1

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 
# In backward propagation we will use y_head that found in forward progation

# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling sondaki bölme işareti değerlerin küçük çıkmasını sağlar

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]# x_train.shape[1]  is for scalingbüyük değer çıkmasın diye sondaki bölme işlemi

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
# Updating(learning) parameters

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    #yukarıdaki listeler sayesinde costları depolayacaz cost2 sayesinde ise hata maliyetin ne kadar düştügüne grafik üzerinden baakcağız

    

    for i in range(number_of_iterarion):#number_of_iterarion for==forward_backward işlemini baştan sonra kaç defa yapacaz ne kadar yaparsa

        #o kadarda çok güzel öğrenir diyebiliriz.

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)#cost görselleştirceğiz ne kadar öğrendigine grafiktan bakacaz

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]#learning_rate= öğrenme hızı

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:#her 10 adımda bir cost2 dizi içrisene cost değerlerine atacak ve her attıgı costunda bir index numarası olacak indexe bak

            cost_list2.append(cost)

            index.append(i)#buraya bak yani bir cost değerine bakmak istersen indexi zaten belii

            print ("Cost after iteration %i: %f" %(i, cost))#sürekli maliyet değerleri yazdırılmış

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)
 # prediction güncellenmiş ve test resimlerine ihtiyacım var

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))
# intialize parameters and layer sizes 29 video matematik

#kod üzerinde bulunan 3 değeri nod sayısını temsil ediyor.

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):# nod sayısını değiştirmek istersen parametre bölümüne nod değişkeni at oradan değiştir direk

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,# küçük bir sayı elde etmek için 0.1 ile çarpıyoruz

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters


def forward_propagation_NN(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2) # A2 aslında y_head değeri



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache

# Compute cost

def compute_cost_NN(A2, Y, parameters):# y_head değeri A2 zaten benim Y Değerini bilmem lazım karşılaştırma yapacam yani label olayı

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost

# Backward Propagation

def backward_propagation_NN(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads



#dw1 costun w1 e göre olan türevi demek

#dZ1 costun z1 e göre olan türevi demek

#Bu kodu açıklamadı adam matematiksel olunca
# update parameters

#dweight1 yukarıda açıkladğım gibi weight nin costa göre trevi yani güncel olan

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"], #önceki weight-öğrenmehızıçarpı*grads

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# prediction

def predict_NN(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation_NN(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #initialize parameters and layer sizes

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)#kaç tane resim kaç tane nod o sayısı belirledik burada



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation_NN(x_train,parameters)

        # compute cost

        cost = compute_cost_NN(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

         # update parameters

        parameters = update_parameters_NN(parameters, grads)

        

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # predict

    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)
# reshaping

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score#Data seti 5 blüyor 4 bölümlük kısmı train 1 bölümlük kısmı test oluyor.

#bu işlemi 4 defa tekrarlıyor

from keras.models import Sequential # initialize neural network library bunun sayesinde direk w,b instialize ediyoruz

from keras.layers import Dense # build our layers library #layerlar ayarlamak için

def build_classifier():

    #Dense layerları inşa etmemizi sağlayan yapı

    #unit=8 ben bir tane layer okuyorum birinci hidden layerimde 8 tane nod var demektir

    #kernel_initializer = 'uniform' weight ler otomatik olarak tanımlıyor initialize ediyor

    #activation = 'relu' 0 dan küçükse 0 dır 0 dan büyükse 1 dir tarzında

    #input_dim = x_train.shape[1] 4096 olacak değişken dimesionları belirtmek gerekiyor yani bir resim kaç pixselden oluşuyor.

    classifier = Sequential() # initialize neural network yapıyı oluşturduk layerları eklemek lazım

    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))#tanh yerine relu kullandık

    #Dense =4 benim 2.leyarim ve 4 tane nod var demek foto da hidden layer varya onun yanına aynı şekilde 4 tane daha nod layer eklendi 37video

    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))

    #output layeri eklendi artık işlemi sollandırıyoruz arık loos ve cost bulmam lazım 

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    #loss ve cost bir aşagıda buluyoruz loss = 'binary_crossentropy' bu logistic reg. yaptıımız yöntemle aynı yöntem

    #normalde learning rate sabit tutuyorduk ancak adamı kullanırsak sabit olmaz adaptiv momentum demektir kendisi değeri atıyor ve duruma göre daha hızlı öğrenir.

    #metrics = ['accuracy']) değerlendirme yaparken accuracy kullanacağımızı söyleidk dogruluk oranı

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier



#epochs number of iteration demek cv=3 bana 3 tane accuracy bul demek cross_val_score bunla bağlantılı işte bu

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)

#cross_val_score buna bak ya machine learning de anlatmış

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()#accuracy ortalamasını al

variance = accuracies.std()#variance bakıyorum dedi geçti bune buna bak !

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))