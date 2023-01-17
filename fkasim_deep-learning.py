# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

 

import warnings #bu yeni güncellemeler geldi uyarıları oldu artık sizinki esiksi gibi.O uyarılar olmasın diye.

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
x_data = np.load('../input/Sign-language-digits-dataset/X.npy') #dataset numpy formatında olduğu için

y_data = np.load('../input/Sign-language-digits-dataset/Y.npy')



img_size = 64 #datasette verilmiş



plt.subplots()

plt.imshow(x_data[260].reshape(img_size,img_size)) #datanın 260.indeksindekiyi 64x64 boyutunda göster demek

plt.axis("off")



plt.subplots()

plt.imshow(x_data[900].reshape(img_size,img_size))

plt.axis("off") #eksenleri kaldırmak için





plt.show()
#Sonra datasetinden 0 ve 1 olan sign resimlerini ve onların labellerini array olarak birleştirecez.

X = np.concatenate((x_data[204:409],x_data[822:1027]),axis=0) #aşağıdan yukarıya



z = np.ones(205)  #bunlara label olusturdum.Direk Y datasından da çekebilirdik.

y = np.zeros(205)



Y=np.concatenate((z,y),axis=0).reshape(-1,1) #(410,) yazmasın diye



print(X.shape) #yani bir resmim 64x64 bir matris. Bu matrisde renklerin numarası 0ile1 arasında graycolor olduğu için

print(Y.shape)
X[1:3] #64x64 matristen 410 tane var.Bu matrisde(64x64) renklerin numarası aslında yani 0 ile 1 arasında değerler greycolor olduğu için
#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.15,random_state=42)



#burda bütün kodu kendim yazacağım için train ve test datamın kaça kaç matrisler oldugunu öğrenmem lazım

numberof_train = x_train.shape[0]

numberof_test = x_test.shape[0]



print(numberof_train,numberof_test)
#Şimdi burası önemli!!Üç boyutlu idi benim resim datalarım bunun için ben bunu iki boyutlu yapacağım.Yani 64x64 artık tek boyut olacak 4096 olarak

#Bunu sadece resim için yapıyoruz çünkü onlar üç boyutlu sadece

x_train_flatten = x_train.reshape(numberof_train,x_train.shape[1]*x_train.shape[2]) #yani 348 adet resim satırlar ve 4096tane feature

x_test_flatten = x_test.reshape(numberof_test,x_test.shape[1]*x_test.shape[2])      #yani 62 tane resim->62 satır 4096 feature



print(x_train_flatten.shape,"\n",x_test_flatten.shape)
#Transpoze işlemi=> Bu şu için ilerde matris çarpımı yapılacak weightler ve feature ların. (a,b)*(b,c) olması gerektiği için ya weightlerin transpozesi

# (a,b)*(b,c) olması gerektiği için ya weightlerin transpozesi ya da resimlerin tranzpozesi alınmalıdır.!(c,b) yi (b,c)ye dönüştürür.



#Yani olay Resimleri sütuna özelliklerini satıra koymak. 1.sütun 1.resim aşagı doğru 4096tane feature böyle...



x_train = x_train_flatten.T

x_test = x_test_flatten.T

y_train = y_train.T

y_test = y_test.T
print(x_train.shape,y_train.shape) #bu şu sütunlar resim yani 1.sütun bir resim ve aşğı doğru onun bütün(4096 tane) feature ları.

                                   #Tek satır(sınıf ayrımı) sütunlar resimlerin indeksi
def sigmoid(x, derivative=False):

    sigm = 1. / (1. + np.exp(-x))

    if derivative:

        return sigm * (1. - sigm)

    return sigm
#Initializing parameters:weights and bias. (logistic regression'da 0.01 ve 0 olarak başlatıyorduk.) Burda dikkat 2 weight ve 2 bias ımız var.

#Burda diğer önemli nokta da Layer size'ı belirlemek.

#Layer size da bu hidden layerdaki node sayısı!! Ben burda buna 3 diyeceğim.

#Burda random bir şekilde verilecek başlangıç parametreler(çeşitlilik olsun diye). Küçük bir sayı olması lazım ki büyük olursa tanh 1 ve ya -1 çıkar. 

#Bunun türevi de yani eğimi 0 çıkacagından model iyi kurulamaz.



def initialize_parameters_and_layer_sizes(x_train,y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0])*0.1,   

                 "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3)*0.1,

                  "bias2":np.zeros((y_train.shape[0],1))}

    return parameters



#burdaki 3te layersize ım benim 3 olacak yani layer size 3e1lik bir matris.O zaman weight matrisim (3,train.shape[0]) olur.

#bu 0.1 > küçültmek için türev 0 cıkmasın diye

#Şimdi ben hidden layer ımı yani 3,1 lik matrisi tekrar weight ve bias larla çarpcam ve output layer ı olustururcam yani (1,1). Ozaman benim weight2 (1,3) olmalı

#Sonra bias ile toplayacam. (çarptım 1,1 olusturdum. bias ta toplam için 1,1 lik olmalı)

#burda eğer layer size(node sayısı) 3 değilde n olsaydı;fonksiyonda n tanımlayıp parameters kısımlarına n yazacaktım.
#Forward propagation feature matrisi ile weight matrisi çarpılır ve biasla toplanır!!

def forward_propagation(x_train,parameters):

    Z1 = np.dot(parameters["weight1"],x_train) + parameters["bias1"]  #(3,4096) * (4096,348) = (3,348) her bir resim için ise 3,1 node olmus oluyor

    A1 = np.tanh(Z1)                                                  #tanh fonksiyonuna sokuyorum ve hidden layer elde edilmiş oluyor. (3,1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]       #(1,3) * (3,1) = (1,1) output layer

    A2 = sigmoid(Z2)                                                  #sigmoid fonksiyonu

    

    cache = {"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2}

    

    return A2,cache



#bu çıkan A2 değerlerim benim y_head değerlerim aslında   
#Loss and Cost Function:ama. elde ettiğmiz y değerli ile (A2) gerçek y değerlerini karşılaştırdık.

def compute_cost(A2,y_train):

    logprobs = np.multiply(np.log(A2),y_train)   #loss function #gerçek Y değerlerim ile çıkan y_head değerlerimin logiritmasıyla çarpıyorum.

    cost = -np.sum(logprobs) / y_train.shape[1]  #cost function # burda total 348 resme bölüp normalize etmiş oluyoruz.

    return cost



#bu bir matematik formül işlemidir. Ben bu cost değerini amacım 0 yapmak ondan buna update işlemlerini yapacam.
# Backward Propagation:Amaç geriye doğru türev alarak weight ve bias ı güncellemek.

#Cok fazla matematiksel işlem: Cost 'un Z2ye W2ye b2ye .. vs bunlara göre türev işlemi

#Burda eğer hidden layer sayısı arttıkça bu formüllerin karmaşıklığı da artar.



#keepdims demek benim diyelim sum işlemim constant bir değer ama keepdims ile array içinde tutuyor

def backward_propagation(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]     #keepdims demek benim diyelim sum işlemim constant bir değer ama keepdims ile array içinde tutuyor

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads

#Update parameters: Burda learning rate i iyi ayarlamak ne cok küçük ne de çok büyük vermemek lazım.

#Learning rate bir hyper parameter. Yani deneyerek bulmak gereken. Ama genelde 0.01 alınıR!

def update_parameters(parameters,grads,learning_rate=0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters



#Olay başlangıçtaki parametremden öğrenme hızım oranındaki costun parametreye göre türevini çıkartıyorum!
#Artık benim modelim hazır.Yani weight ve bias döngümü tamamladım.Sıra Prediction kısmında

#Prediction with learnt weight and bias



def prediction(parameters,x):

    A2,cache = forward_propagation(x,parameters) #forward ile Yhead değerlerimi buluyorum.Sonra bu değerlerim trashold(0.5)'a göre 1veya0 olacak'

    

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

    
#ANN Model

#num_iteration bu öğrenme süresi ne kadar uzun sürsün ne kadar adım yapalım. Mesela bu türev alma işlemi var ya kaç kez güncelleyelim parametreleri



def two_layer_artificial_neural_network(x_train,y_train,x_test,y_test,num_iterations):

    cost_list = []

    index_list = []

    

    #initialize parameters

    parameters = initialize_parameters_and_layer_sizes(x_train,y_train)

    

    #number of iteration kadar parametrelerimi güncelleme işlemini yapacam.

    for i in range(0,num_iterations):

        #forward propagation

        A2,cache = forward_propagation(x_train,parameters)

        

        #compute cost

        cost = compute_cost(A2,y_train)

        

        #backward propogation

        grads = backward_propagation(parameters, cache, x_train, y_train)

        

        #update parameters

        parameters = update_parameters(parameters, grads,learning_rate=0.01)

        

        #Bu şu ben diyecem ki 100 iterasyon git ve en son 100. defa parametreleri güncelle.

        #Her 100 adımda bir bana cost değerlerini göstermesini istersem

        

        if i%100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print("Cost after %i iteration : %f" %(i,cost))

            

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost Value")

    plt.show()



    #Buraya kadar Modeli kurduk, en son güncellenmiş parametre değerlerini belirledik ve cost değerlerini plot ettirdik. Sıra Prediction ve Accuracy

    #Predicion

    

    y_prediction_test = prediction(parameters,x_test)

    #Sonuç, ACCURACY #bu accuracy formulü!

    

    # Print train/test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters

        

        
parameters = two_layer_artificial_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)
#Keras da reshaping gerekli transpoze almamız lazım

x_train,x_test,y_train,y_test = x_train.T,x_test.T,y_train.T,y_test.T
x_train.shape[1] #yani x verisi, satırda resim sütunda ise feature lar normal kısım!
#Evaluating ANN with Keras



from keras.wrappers.scikit_learn import KerasClassifier  #bir datayı classify ederken bu method kullanılır

from sklearn.model_selection import cross_val_score      #cross validation score(birden fazla train test olusturup ortalama accuracy i bulmak)

from keras.models import Sequential                      #initialize parameters

from keras.layers import Dense                           #construct the Layers (layerları olusturmada)



#neural network umu olusturacak yapının methodu

def build_classifier():

    classifier = Sequential()                            #intialize neural network. Benim artık bir yapım var.ve ben bu yapıya artık layerlarımı ekleyecem.

    

    #forward propagation

    classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation = "relu",input_dim = x_train.shape[1]))   #1.hidden layer

    classifier.add(Dense(units = 4, kernel_initializer = "uniform", activation = "relu"))                                #2.hidden layer

    classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))                             #output layer

    

    #compute loss and cost function, backward propagation

    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

    

    return classifier



#Benim artık classifier ım build edildi. Sıradaki işlem classifier ımı çağırmak olacak. Bunu da KerasClassifier methodu ile yapcam

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100) #build_fn: classifierımı buil ettiğim fonksiyonum, epochs:number of iterations



#Artık yapıyı kurdum. Fonksiyonumu çağırdım ve Şimdi sırada ise datamı eğitmek.Bunu cross validation ile yapcam(Yani birden fazla train ve test datası olusturcak)

#(birden fazla accuracy verecek! ve bunların mean değerini alacam bu daha effective bir sonuc olmuş olacak!

accuracies = cross_val_score(estimator = classifier, X = x_train, y=y_train, cv = 3) #cv=3 demek bana 3 tane accuracy bul demek.

print(accuracies)

mean = accuracies.mean()

variance = accuracies.std()

print("Mean Accuracy:"+str(mean))

print("Variance:"+str(variance))

    