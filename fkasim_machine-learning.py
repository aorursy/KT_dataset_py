# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#burda excel ile yapılınca csv dosyası noktalı virgül ile ayırıyo bu da kaggle da sıkıntı yaratıyo. Onu virgülle ayırmalıyız

df = pd.read_csv("../input/regressiondata/linear-regression.csv")
df

#görüldüğü gibi noktalı virgül sıkıntı yaratıyo
#bunu sep metodu ile engelleyebiliriz.

df = pd.read_csv("../input/regressiondata/linear-regression.csv",sep=";")
df
plt.scatter(df.deneyim,df.maas)

plt.xlabel("deneyim")

plt.ylabel("maas")

plt.show()
#residual(hata) = y - y_head(prediction)  ---> çift taraflı kareler alınması lazım negatiflikten kurtarmak için

#sum(residual^2) bunun en min değeri the best fitting line

#MSE (mean squared error) = sum(residual^2) / n(sample)

#the goal---> min(MSE)
#bu machine learning algoritmaları için kullanıcalak kütüphane sklearn. Bunun içinde birsürü machine learn algoritması var.

from sklearn.linear_model import LinearRegression  #bu kütüphaneden linear regresyon algoritmasını import ediyoruz.

linear_reg = LinearRegression() #böyle bir model kuruyoruz.

#daha sonra datasetimizden verieleri almalıyız ama pandas olarak almamız sıkıntı yaratabilir onun için numpy olarak almalıyız yani bir array olarak almalıyız

x = df.deneyim

type(x)
#görüldüğü gibi type pandas seri olarak gözüküyor ama bunu array olarak almamız gerekiyor.

x = df.deneyim.values

x
x.shape  #burda shape 14, gözüküyor. bunu sklearn kütüphanesi kabul etmiyor bunun için reshapelememiz gerekmektedir.
x = df.deneyim.values.reshape(-1,1)
x.shape
#aynı işlemi y için de yapmamız gerekmektedir.

y = df.maas.values.reshape(-1,1)
y
#şimdi fit dogrusunu buldurabiliriz. bunun da kurdugumuz model ve fit() metodu ile gerçekleştirmeliyiz.

linear_reg.fit(x,y)
#fit ettiğimiz tahmini regresyon dogrumuzun b0(intercept) ve b1(coefficent) değerlerini görebiliriz.

#b0 = linear_reg.predict(0)   # xin 0 oldugu nokta b0 dır.

b0 = linear_reg.intercept_  #b0 intercept ile bulunabilir.

b0
b1 = linear_reg.coef_  #eğim slope

b1
#regression dogrum --->  maas = 1664 + 1138*deneyim

print(linear_reg.predict(11))
#bu hatada sunu diyor tek boyutlu birsey yazıyosun diyo yani 11. ama 2 boyutlu bişi yazman gerekiyor bunu da köşeli parantezle iki boyutlu yapmalısın bir array oldugu için!!

maas = linear_reg.predict([[11]])

maas
#bu problemin diğer çözümü ise sudur

b0 = 12

b0 = np.array(b0).reshape(-1,1)

b0 = linear_reg.predict(b0)

b0
#şimdi fitting line ı mızı grafikte görelim. onun için bir numpy ile array olusturcam liste olustursam galiba for döngüsü kullanmam gerekecek onun için array.

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1) #bunlar benim x değerlerim deneyim yılları yani hata almayalım diye reshapeliyoruz!

#şimdi tahmini dogru için değerleri bulcam

y_head = linear_reg.predict(array)   #burdada yine aynı hatayı almamak için 2 boytlu yapmak gerek.

plt.scatter(array,y_head)  #bu scatter şeklinde noktaları gösterir.

plt.plot(y_head)  #bu ise noktaları çizgi şeklinde gösteririr!

plt.show()
plt.scatter(x,y)

plt.plot(array,y_head,color="red")

plt.show()
dataset = pd.read_csv("../input/multipleregression/multiple-linear-regression-dataset.csv",sep=";")
dataset

#önemli olan maas=dependent variable , yas ve deneyim = independent variable olması lazım! Dikkat et independentlar arasında korelasyon olmasın!
from sklearn.linear_model import LinearRegression  #burda yapcagımız işlemler aynı sadece feature sayısı artıyo
#x değişkenimiz yas ve deneyim olacak

x = dataset.iloc[:,[0,2]].values   #burda diyo ki tüm satırları al ve sadece 0. ve 2. sütunu al! ve tabi onların değerlerini. görüldüğü gibi reshape etmeye gerek kalmadı çünkü 1 için yazmıyo sadece

y = dataset.maas.values.reshape(-1,1)
multiple_reg = LinearRegression()   #bu metodu objeye ceviriyoruz

multiple_reg.fit(x,y)               #x ve y ye göre bir fit dogrusu olusturuyoruz.
b0 = multiple_reg.intercept_

print(b0)

b0 = multiple_reg.predict([[0,0]])  #burda [[0]] diyemeyiz çünkü multiple reg yaptıgımızdan dolayı b1 ve b2 var onların ikisininde 0 olması gerek

print(b0)
print("b1,b2 =",multiple_reg.coef_)

#b2 yani yaş katsayısı - çıktı yaş arttıkca maaş azalır sonucunu verir.
array = np.array([[10,35],[6,24]])

y_head = multiple_reg.predict(array)

y_head
plt.scatter(x,y)   #aynı boyutta olma hatası verdi!

plt.show()
df = pd.read_csv("../input/polinomial/polynomial-regression.csv",sep=";")
df
plt.scatter(df.araba_fiyat,df.araba_max_hiz)

plt.show()
#bu datasetini linear regression modeline göre yaparsak 

from sklearn.linear_model import LinearRegression



lr = LinearRegression()



x = df.araba_fiyat.values.reshape(-1,1)

y = df.araba_max_hiz.values.reshape(-1,1)



lr.fit(x,y)   #best fitting



y_head = lr.predict(x)



plt.scatter(x,y)

plt.plot(x,y_head,color="red")

plt.show()

#yukarıdaki grafiktede görüldüğü gibi olusan predict fitting regression dogrusu çok iyi değil mesela örnek verecek olursak

lr.predict([[10000]])   #arabanın fiyatını 10milyon yaparsak hızını 870km gösteriyo bu da mümkün değildir.
#arabanın fiyatı arttıkca hızı bi yerde sabit kalıyo daha yükselmiyo ondan parabolic bir denklem ortaya cıkıyor

#burda data setimnizde bizim xkareli bir featuremız yok bundan dolayı bizim bir xkareli feature olusturmamız gerek. İlk dereceyi 2 alacaz bu dataseti için

#polynomial bir feature ı sklearn kütüphanesinden olusturabiliyiyoruz!!

from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree=2) #metodunu tanımlıyoruz ve kacıncı dereceye kadar burda vermemiz gerekiyo

x_polynomial = polynomial_reg.fit_transform(x)  #burda araba fiyatımız olan feature ın karesi feature nı fit_transform metodu ile yapıyoruz
#artık xkareli feature mızı olusturduk geri kala işlemler linear regressiondaki işlemlerin aynınısıdır. Önemli olan kac dereceli polinom yarattıgımız 

lr2 = LinearRegression()

lr2.fit(x_polynomial,y)   #artık x_polynomial feature ını y ye göre fit edebiliriz. burdaki denklem su oluyo=y=b0+b1*x+b2*x^2

#şimdi grafik üzerinde çizdirebiliriz

y_head2 = lr2.predict(x_polynomial)



plt.scatter(x,y)

plt.plot(x,y_head,color = "green",label= "linear")  #burda ilk linear egression ile yaptıgımız tahmini dogru

plt.plot(x,y_head2,color="red",label="polynomial")       #daha sonra polynomial reg ile yapılan

plt.legend()  #label kısmı lejant için yazılır

plt.show()
#peki ben bu modeli daha da iyileştirebilirmiyim?

#Tabiki.Sadece Polinom derecesini artırarak modeli iyileştirebilirim. Yani dereceyi ben 4 yaparsam x^4 e kadar x değerleri olusur

#bunun sonucunda model daha da iyileşir ! Hemen yapalım

from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree=4) #metodunu tanımlıyoruz ve kacıncı dereceye kadar burda vermemiz gerekiyo. İYileşme için 4.dereceye kadar x olusur

x_polynomial = polynomial_reg.fit_transform(x)  #burda araba fiyatımız olan feature(x) ın 4.dereceye kadar feature değerleri olusur. burda fit_transform metodu



#artık x^4 feature mızı olusturduk geri kala işlemler linear regressiondaki işlemlerin aynınısıdır. Önemli olan kac dereceli polinom yarattıgımız 

lr2 = LinearRegression()

lr2.fit(x_polynomial,y)   #artık x_polynomial feature ını y ye göre fit edebiliriz. burdaki denklem su oluyo=y=b0+b1*x+b2*x^2



y_head3 = lr2.predict(x_polynomial2) #4.dereceye kadar olan x değerlerinin tahmini hız değerleri

plt.scatter(x,y)

plt.plot(x,y_head,color = "green",label= "linear")  #burda ilk linear egression ile yaptıgımız tahmini dogru

plt.plot(x,y_head2,color="red",label="polynomial2")       #daha sonra polynomial reg ile yapılan 2.dereceli

plt.plot(x,y_head3,color="black",label="polynomial4")       #daha sonra polynomial reg ile yapılan 4.dereceli

plt.legend()  #label kısmı lejant için yazılır

plt.show()
#burda belli bir fiyatın tahmini hız değerini bulabilmek için polinomda şöyle yapıyoruz. biz fit dogrusunu 4.dereceye göre fit ettiysek

#o zmaan x feature değerleri sırasıyla [x^0,x^1,x^2,x^3,x^4] dizisi olmus oluyo.

lr2.predict([[1,1000,1000000,1000000000,1000000000000]]) #burda 1milyon tl olan aracın tahmini hızı 
#dataseti tribündeki kategorilerin fiyatları

df = pd.read_csv("../input/decison-tree/decision-tree-regression-dataset.csv",sep=";")

df

#burdaki sütun isimlerini none yapabilirz yan index yapabilirz.
df = pd.read_csv("../input/decison-tree/decision-tree-regression-dataset.csv",sep=";",header=None) #header ile!!!

df
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
plt.scatter(x,y)

plt.show()
#Simdi regresyon işlemi

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()   #random sate = 0

tree_reg.fit(x,y)  #ağac yapımı olusturdum
tree_reg.predict([[5]])
y_head = tree_reg.predict(x)  #burda x bi array zaten

plt.scatter(x,y)

plt.plot(x,y_head,color="red")

plt.show()
#yeni bi array olusturursak

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)  #diyorum ki x dizisini düzenle en küçük ile en büyüğü arasında 0.01 artarak bir dizi olustur.

x_
#tekrardan görselleştirirsek!

y_head = tree_reg.predict(x_)  #burda x bi array zaten

plt.scatter(x,y)

plt.plot(x_,y_head,color="red")

plt.show()
import sklearn.datasets as datasets

iris=datasets.load_iris() #burda iris datasetini array olarak yüklüyor.

df=pd.DataFrame(iris.data, columns=iris.feature_names) #burdada diziyi dataframe ceviriyor!!

iris
df
y = iris.target

y
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier()

dtree.fit(df,y)  #fit ediyoruz!
#Now that we have a decision tree, we can use the pydotplus package to create a visualization for it.

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())   #burda pydotplus modülü yokmus jupyter notebook ile yapılabilir.
#Decision Tree deki aynı dataseti kullanılacak. Sahaya yakın yer en pahalı en uzak yer en ucuz

df = pd.read_csv("../input/decison-tree/decision-tree-regression-dataset.csv",sep=";",header=None) #header ile!!!
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 42)  #burda ben random forest ın içinde kac tane tree kullanacam bunu vermem lazım

#random state ise ben n sample lara ayırırken datamı her zaman aynı ayırsın diye 42 yazarım cıkan sonuc bi önceki ile aynı cıksın. istatistikte vardıya

rf_reg.fit(x,y)
rf_reg.predict([[7.6]])
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators = 100, random_state = 5)  #ramdom_state i değiştirirsem

rf_reg.fit(x,y)

rf_reg.predict([[7.6]])

#görüldüğü gibi verdiği tahmin aynı cıkmadı!!!
#ben bunu şimdi görselleştircem decison tree de yaptıgım gibi

x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head = rf_reg.predict(x_)



plt.scatter(x,y)

plt.plot(x_,y_head,color="red")

plt.show()
#Random Forest örneği için Evaulation

df = pd.read_csv("../input/decison-tree/decision-tree-regression-dataset.csv",sep=";",header=None) #header verinin sütun baslıklarını ataR!
df  #index numarası verdi görüldüğü gibi baslıga

    #kategoriye göre bilet fiyatları
x = df.iloc[:,0].values.reshape(-1,1)

y = df.iloc[:,1].values.reshape(-1,1)

plt.scatter(x,y)

plt.show()
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators = 100, random_state = 42) #burda sınıf sayısını ve randomluk derecesini(yani bi daha run ettiğimde aynı randomlukta çalıştır) vermeyi unutma!

rf.fit(x,y)
rf.predict([[3.5]])
y_head = rf.predict(x)

y_head   #elimizde bir x data sı ile fiyat olan y leri tahmin ettik şimdi tahminimiz ne kadar dogru ona bakcaz.
#Bunun için sklearn kütüphanesinde baska bir metodu kullanacaz.

#Bu modelleri değerlendiren metodlar metrics diye gecer.

from sklearn.metrics import r2_score

print("R_score =",r2_score(y,y_head))  #kullanımı gayet basit sadece gerçek değerler,tahmini değerler



#1'e yakın olması iyi bir prediction oldugu anlamına gelir. Bunu multiple regresyonda da yapılır. Önemli olan gerçek y, tahmini y değer karsılastırılması

#zaten bu metod bizim direk formulu uyguluyo! R_square = 1- (SSR/SST)
#Evaulation Regression with Rscore in Linear Regression

df = pd.read_csv("../input/regressiondata/linear-regression.csv",sep=";") #deneyime göre maas tahmini

df
x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)
lr.predict([[15]])
y_head = lr.predict(x)

y_head
plt.scatter(x,y)

plt.plot(x,y_head,color="red")

plt.show()
#Evaulation

from sklearn.metrics import r2_score

r_score = r2_score(y_true=y,y_pred=y_head)

print(r_score)
df = pd.read_csv("../input/regressiondata/linear-regression.csv",sep=";") #deneyime göre maas tahmini

x = df.deneyim.values.reshape(-1,1)

y = df.maas.values.reshape(-1,1)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000,random_state=42)

rf.fit(x,y)
print(rf.predict([[15]]))

y_head = rf.predict(x)
plt.scatter(x,y)

plt.plot(x,y_head,color="red")

plt.show()

#Görüldüğü gibi daha spesifik bir tahmin doğrusu!
from sklearn.metrics import r2_score

rscore = r2_score(y_true=y,y_pred=y_head)

print(rscore)
#Şimdi ilk bunları bir tümör dataseti üzerinde herseyi elimizle yazarak kodlayacaz. Daha sonra sklearn kütüphanesini kullanacağız!!

df = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

df
df.info()
#Bu data da tümörün featurelara göre diagnosis sınıfı verilmiş. İyi huylu mu kötü huylu mu diye

#id ve unnamed:32 sütunu tümörü sınıflandırma için gerekmez. Bundan dolayı bunları drop edeceğim.

df.drop(["Unnamed: 32","id"],axis=1,inplace = True)  #axis=1 sütunu komple demekti(axis=0 ise row içindi). inplace ise daraframe eşitle demekti

df
df.columns.values
#Şimdi classify yaparken diagnosis object gözüküyo. Ama ya integer ya da categorical bir değer olması gerekiyor.

#ilk integer'a göre cevirecez yani 0 ve 1 değişkenine döndürcez.

df.diagnosis = [ 1 if i == "M" else 0 for i in df.diagnosis ]  #M(kötü huylu ise) 1, B ise 0 yapıyoruz.
print(df.diagnosis.values)

df.diagnosis.count()
#burda benim y eksenim yani sınıflarım diagnosis, x eksenim ise diğer bütün feature lar

y = df.diagnosis

x_data = df.drop(["diagnosis"],axis=1)
#Diğer en önemli nokta da şu-> featurelar da 2500 değeri olup 0,00032 bilmem değeri olan da var. Bu durum büyük olan diğer feature ı etkisiz bırakabilir.

#Bundan dolayı bütün değerleri 0 ile 1 arasında bir değer yapacagım ki birbiri arasında etkileşim olmasın modelim bozulmasın. Buna Normalization denir.

#Normalization(featureların birbirine üstünlük saglamamaları gerek.)

#Formül => (x-min(x)) / (max(x)-min(x))



x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values  #values demek numpy array e cevirmektir.
a = pd.DataFrame(x)

a.describe()  #görüldüğü gibi 0-1 arasında bütün değerler dagılmıstır.
#Train Test Split

#Bu bölüm elimde bir data var ben bu datayı eğitecem ve modelimi kuracam ama aynı zamanda modelim dogru calısıyor mu diye test etmem de gerekecek.

#Bundan dolayı elimdeki data nın %80 ile train yapacam. Geri kalan %20lik data mı ise test için kullancagım!

#Bu ayırma işlemini cok güzel yapan sklearn kütüphanemiz var!



from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42) 

#bu metot bize 4 adet çıktı veriyor sırasıyla.(random_state aynı randomlık içindi hani tekrar calıstırdıgımızda aynı değerleri versin diye)

#test boyutunu %20 aldık.
x_test  #görüldüğü gibi 110 tane sample var. Ve bu x_test in sırası y_test in ki ile aynıdır.
#Şimdi hani konu anlatımında pxel ler yukarıdan asagıya idi(yani featurelar) ve resimler soldan saga matriste idi yani(farklı sampler)

#Burda ters bunun için transpoze alacağız.

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



print("x_train:",x_train.shape)

print("x_test:",x_test.shape)

print("y_train:",y_train.shape)

print("y_test:",y_test.shape)



#yani benim 30tane pixelim yani feature ım var ve benim 455 tane eğitelecek örneğim ve 114 tane test edilecek sample var.
#Initializing Parameters and Sigmoid Function

#Neydi hani baslangıcta benim her pixelimin(yani featuremın) bir baslangıc weight i vardı.Bunları baslangıcta 0.01 alıyorduk.(ama başka teknikler ilerde deep'te)

#Sonra bunları toplayım bias ı ekliyorduk baslangıc biası da 0 kabul ediyorduk.

#Daha sonra cıkan değeri 0 ie 1 arasında bir değer versin diye sigmoid functiona sokuyorduk ve eğer 0.5ten büyükse sonucu 1 alıyorduk.Yani kötü huylu



def initialize_weight_and_bias(dimension):   

#bu dimension su: hani her pixel için bir weight vardı ya burda da her feature için bir weight olmalı. 30 tane feature oldugu için 30 tane weight olacak.

    

    w = np.full((dimension,1),0.01)  #aynı np.ones veya np.zeros gibi np.full da istediğimiz matrisi yaratmamızı saglar.Yani 30'a 1lik matris ve bütün w'ler 0.01

    b = 0.0

    

    return w,b

#Sigmoid Function

#Şimdi sırada sigmoid funtion ın formülasyonunu yazacaz. İnternette hemen bulabilirsin.

#formül => f(x) = 1 / (1 + e^(-x))



def sigmoid_func(z):

    

    y_head = 1 / (1 + np.exp(-z))   #e üssü demek np.exp(x) demektir!!

    return y_head



#Zaten bu cıkan y_head değeri ile x_test değerlerini girerek y_test değerleri ile karsılastıracagız.

    
#ex

sigmoid_func(6)
#Forward and backward propagation

def forward_bacward_propagation(w,b,x_train,y_train):

    #forward propagation

    z = np.dot(w.T,x_train) + b  #burda klasik matris carpımı yapıyoruz. (a,b)(b,c) matris çarpımı olur. Ondan weight'in transpozesi alınır.

    y_head = sigmoid_func(z)

    loss = -(1-y_train)*np.log(1-y_head) - (y_train*np.log(y_head))  #bu da loss function ın formulu idi.

    cost = np.sum(loss) / x_train.shape[1]  #cok abartı cıkmasın diye ortalama alıyoruz. 'for scaling' yani ölçülendirme için

    

    #backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #türev formülü bu sabit bir matematiksel ifade-türev demek eğim demek

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}  #bu yeni weight ve bias ı sözlük içinde depoluyoruz.

    return cost,gradients

#Update

#Şimdi sırada weight ve bias ları güncelleme yaparak en iyi değerlerini ve en az cost değerini bulma yapacaz.



def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list = list()

    cost_list2 = list()

    index = list()

    

    #updating parameters is number_of_iterations times 

    for i in range(number_of_iteration): #biz her bir forward ve backward yapmamız bir iteration dır.

        cost,gradients = forward_bacward_propagation(w,b,x_train,y_train) #bunun fonksiyonunu yazmıstık

        cost_list.append(cost)  #bütün iterasyon sonucu cıkan cost u listeme atıyorum.

        

        #update kısmı=aslında benim modelimi train etmek demektir!!

        #her iterasyonda yeni değerleri w,b ye esitliyoruz

        w = w - learning_rate*gradients["derivative_weight"] #burda gradients yeni w ve bias değerleri sözlükte depoladıgımız

        b = b - learning_rate*gradients["derivative_bias"]

        #bu formül vardı hani leraning rate ile.öğrenme hızı yavas olsa uzun sürer fazla versek hiç öğrenemeyebilirz.

        

        if i %10 ==0:

            cost_list2.append(cost)   

            #bunu yapmamın amacı tamamen görünüş ile alakalı. Çok fazla iterasyonda o kadar cost cıkacak 10da bir yazdırırsam daha güzel gözükür.

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))  #bu yazımı öğren aynı formatlama gibi

    

    parameters = {"weight":w,"bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    

    return parameters,gradients,cost_list    #gradients=türevlenmis hali w ile b nin
#Prediction 

def prediction(w,b,x_test):

    z = sigmoid_func(np.dot(w.T,x_test) + b)

    Y_prediction = np.zeros(1,x_test.shape[1])  #bir matris olusturuyorum (1,114 lük) karsılastırma için

    

    #if z is bigger than 0.5, our prediction is 1(y_head=1) kötü

    #if z is lower than 0.5,our prediction is 0 (y_head=0) iyi

    

    for i in range(z.shape[1]):   #daha sonra cıkan z değerleri ile kosullu durumla y_prediction matrisimi dolduruyorum!!

        if z[0,i] <=0.5:   #treshold kısmı 0.5

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    

    return Y_prediction
#Implemeting Logistic Regression

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):

    #initialize

    dimension = x_train.shape[0] #that is 30

    w,b = initialize_weight_and_bias(dimension)

    #sırada forward ve backward var ama ben ayrı ayrı yazmak yerine zaten update metodunun içinde kullandım forwardbackward ı

    #do not change learning rate

    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,number_of_iteration)

    

    #prediction

    y_prediction_test = prediction(parameters["weight"],parameters["bias"],x_test)

    

    print("Test Accuracy {} %".format(100-np.mean(np.abs(y_prediction_test - y_test))*100))

    
#Result

logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,number_of_iteration=100)
# %% read csv

data = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

data.drop(["Unnamed: 32","id"],axis=1,inplace = True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

print(data.info())



y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)



# %% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



# (x - min(x))/(max(x)-min(x))



# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



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



#%% Updating(learning) parameters

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

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



#%%  # prediction

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



# %% logistic_regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)    



data = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

data.drop(["Unnamed: 32","id"],axis=1,inplace = True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

print(data.info())



y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)
# %% normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42) 
#%% sklearn with LR

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

print("test accuracy {}".format(lr.score(x_test,y_test)))  #accuracy aynı R2score gibi tahminin doğrulugunu ölçüyor.
df = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv") #aynı kanser datası üzerinde çalısılcak

df.tail()

#Malignant = M kötü huylu

#Benign = B iyi huylu tümör
#Id ve unnamed featurelarından kurtulcam.çünkü gereksiz featurelar

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.head()
#Şimdi datamı iyi huylu ve kötü huylu olarak ikiye ayırcaam

M = df[df.diagnosis == "M"]

B = df[df.diagnosis == "B"]

M.info()
B.info()
#M ve B yi 1,0 şeklinde cevirecem. Çünkü biz class label larımızı string istemiyoruz ya integer ya da categorical

df.diagnosis = [1 if i=="M" else 0 for i in df.diagnosis]

df.head()
#Scatter iyi huylu ve kötü huylunun radius_mean ile area_mean karsılastırılması

plt.scatter(M.radius_mean,M.area_mean,color="red",label="Kotu")

plt.scatter(B.radius_mean,B.area_mean,color="green",label="Iyi")

plt.legend() #lejant

plt.show()
#Başka feature ları scatter yapalım

plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Kotu",alpha = 0.4)  #alpha saydamlık oranı!

plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Iyi",alpha= 0.4)

plt.xlabel("radius_mean")

plt.ylabel("texture_mean")

plt.legend() #lejant

plt.show()
#Data mızı x ve y olarak ayırıyoruz. value demek array a cevirir.

y = df.diagnosis.values  #datamızın iyi huylu mu kötü huylumu sonuc değerleri

x_data = df.drop(["diagnosis"],axis=1)  #diagnosisi cıkarırsam geriye kalan x matrisi olur

y
#Normalization

#En önemli şey= normalization çünkü noktalar arası mesafe bulanacak aynı derecede değerlendirmeliyiz.

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
#Train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 42 )
y_train
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3 )  #n_neighbor = k

knn.fit(x_train,y_train)  #modeli eğitiyoruz

prediction = knn.predict(x_test) #x_test değerlerinin y değerlerini tahmin et

prediction
#Şimdi K=3 ken accuracy e bakalım

print("K={} iken accuracy: {}".format(3,knn.score(x_test,y_test)))
#Find k value

k_value = []

accuracy = []

for i in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train,y_train)

    score = knn2.score(x_test,y_test)

    k_value.append(i)

    accuracy.append(score)

for i,j in zip(k_value,accuracy):

    print(i,j)
#Find K value for Max accuracy

plt.plot(range(1,15),accuracy,color = "blue")

plt.xlabel("K value")

plt.ylabel("Accuracy")

plt.show()
max_deger = 0 #accuracy

for i in range(1,200):

    knn = KNeighborsClassifier(n_neighbors=i) 

    knn.fit(x_train, y_train)

    score = knn.score(x_test,y_test)

    

    if score > max_deger:

        max_deger,k = score,i  

    else:

        continue

print(max_deger,k)
df = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.diagnosis = [1 if i=="M" else 0 for i in df.diagnosis]
y = df.diagnosis.values

x_data = df.drop(["diagnosis"],axis=1)
#normalization

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.svm import SVC

svm = SVC(random_state=42)

svm.fit(x_train,y_train)

print(svm.score(x_test,y_test))
#konu anlatımındaki o çemberi similatiry_range ile belirliyoruz. Ama hoca anlatmamıs kod da ona bakarsın!

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print(nb.score(x_test,y_test))
df = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.diagnosis = [1 if i=="M" else 0 for i in df.diagnosis]

y = df.diagnosis.values

x_data = df.drop(["diagnosis"],axis=1)



x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = 42)  #bunun içinde de random_state algoritması var. Her zaman aynı randomluk için bunu da yaz.

dt.fit(x_train,y_train)

dt.score(x_test,y_test)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 42) #n_estimator = tree sayısı

rf.fit(x_train, y_train)

rf.score(x_test,y_test)
#Random Forest Classfi algoritmasını kullancam bu örnekte

df = pd.read_csv("../input/ninechapter-breastcancer/breastCancer.csv")

df.drop(["id","Unnamed: 32"],axis=1,inplace=True)

df.diagnosis = [1 if i=="M" else 0 for i in df.diagnosis]

y = df.diagnosis.values

x_data = df.drop(["diagnosis"],axis=1)

x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data))



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=42)



from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100,random_state = 42) #n_estimator = tree sayısı

rf.fit(x_train, y_train)

print("Accuracy: ",rf.score(x_test,y_test))

#Confusion Matrix i elde edebilmek için y_pred ve t_true değerlerine ihtiyacım var.

y_pred = rf.predict(x_test)

y_true = y_test



#Confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

print(cm)
#Confusion matrix visualization

#Heat map yapcaz!!!

import seaborn as sns

f, ax = plt.subplots(figsize = (5,5))  # bu plot'umun boyutunu ayarlar!!

sns.heatmap(cm,annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f", ax = ax) 

#annot= matrisin sayılarını grafikte yazdırmak için(false ya da annot yazmazsak sayılar yazmaz)

#fmt = ondalık basamaklar için kullanılır.(.0f demek ondalık basamak yok)

#eksenim ax tir.

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

#Burda datasetimizi kendimiz hazırlayacaz! Ben cluster sayımı 3e göre hazırlayacam.Bakalım K-means algoritması çözecekmi dogru!

#Şimdi 2 feature ı olan ve 3 sınıflı bir data



#class 1

x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)



#class 2

x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)



#class 3

x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)

                                         #bu concanate methodu sadece yukarıdan asagı birleştirme işlemi yapar.   

x = np.concatenate((x1,x2,x3),axis = 0)  #aynı pd.concat gibi birleştirme işlemi. axis 0 demek yukarıdan asagı birleştir.

y = np.concatenate((y1,y2,y3),axis = 0)



sozluk = {"x":x,"y":y}

df = pd.DataFrame(sozluk) #Dataframe olusturma

df
#diğer bir data olusturma

z=np.column_stack((x,y)) #column_stack metodu ise sütüunları yanyana birleştirir.

data = pd.DataFrame(z)

data.rename(columns={0:"x",1:"y"},inplace = True)

data
f,ax=plt.subplots(figsize=(10,6))

plt.scatter(df.x,df.y,color="red",alpha=0.5)

plt.xlabel("x")

plt.ylabel("y")

plt.show()
f,ax=plt.subplots(figsize=(10,6))

plt.plot(x1,y1,label="class1",color="red")

plt.plot(x2,y2,label="class2",color="green")

plt.plot(x3,y3,label="class3",color="black")

plt.xlabel("x")

plt.ylabel("y")

plt.legend()

plt.show()
#Aslında K mean için benim böyle bir datam var. Bakalım K means ile nasıl cözecez

f,ax=plt.subplots(figsize=(10,6))

plt.scatter(df.x,df.y,color="black",alpha=0.5)

plt.xlabel("x")

plt.ylabel("y")

plt.show()
# Kmeans

#ilk önce K değeri,bunun için de wcss metriği

from sklearn.cluster import KMeans

wcss = []

for k in range(1,16):

    kmeans = KMeans(n_clusters = k)

    kmeans.fit(df)

    wcss.append(kmeans.inertia_)   #inertia_ methodu her bir k değeri için wcss değerini döndürür.

plt.plot(range(1,16),wcss,color="purple")

plt.xlabel("K_values")

plt.ylabel("WCSS")

plt.show()

    
#K=3 için modelim

kmeans2 = KMeans(n_clusters = 3)

clusters = kmeans2.fit_predict(df)  #burda clusterlarımı fit_predict methodu ile. Modeli kur ve kümeleri olustur tahimn et demek



pd.DataFrame(clusters) #görüldüğü gibi üç gruba ayırdı ve gruplara 0,1,2 numaralarını verdi.

#Şimdi bu grupları datama ekleyeyim

df["label"] = clusters

df
#Şimdi de görselleştirelim bakalım doğrumu sınıflandırmış. 

#Datanın saf hali ve tahmin edilen sınıflara göre görselleştirip karsılastıralım.

#Saf hali

plt.scatter(df.x,df.y)

plt.show()
#Labellara göre

plt.scatter(df.x[df.label==0],df.y[df.label==0],color = "red")  #x ve y eksenleri label ları 0 olanlar

plt.scatter(df.x[df.label==1],df.y[df.label==1],color = "blue") #x ve y eksenleri label ları 1 olanlar

plt.scatter(df.x[df.label==2],df.y[df.label==2],color = "green")#x ve y eksenleri label ları 2 olanlar



#!!!Clusterların centroidlerini gösterme methodu!!!!!!!!

plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color="yellow")

#Bu iki boyutlu bişi olduğu için x,y olarak cıktı alcaz. : --> bütün centroidleri al x ve y koordinatları

plt.show()
#Kmeans de yaptıgım gibi numpy ile dataset olusturcam.

#Burda datasetimizi kendimiz hazırlayacaz! Ben cluster sayımı 3e göre hazırlayacam.Bakalım HC algoritması çözecekmi dogru!

#Şimdi 2 feature ı olan ve 3 sınıflı bir data



#class 1

x1 = np.random.normal(25,5,1000)

y1 = np.random.normal(25,5,1000)



#class 2

x2 = np.random.normal(55,5,1000)

y2 = np.random.normal(60,5,1000)



#class 3

x3 = np.random.normal(55,5,1000)

y3 = np.random.normal(15,5,1000)

                                         #bu concanate methodu sadece yukarıdan asagı birleştirme işlemi yapar.   

x = np.concatenate((x1,x2,x3),axis = 0)  #aynı pd.concat gibi birleştirme işlemi. axis 0 demek yukarıdan asagı birleştir.

y = np.concatenate((y1,y2,y3),axis = 0)



sozluk = {"x":x,"y":y}

df = pd.DataFrame(sozluk) #Dataframe olusturma

df
plt.scatter(x1,y1)

plt.scatter(x2,y2)

plt.scatter(x3,y3)

plt.show()
#Dendogram

#Dendogram için farklı bir kütüphane kullanacağız. scipy kütüphanesi

from scipy.cluster.hierarchy import linkage, dendrogram  #bu import ettiğimiz linkage, hieararchy algoritması için



merg = linkage(df,method = "ward") #K means ta method olarak wcss(en min uzaklıklar toplamı için) kullanıyorduk.

                                   #ward ise clusterlarımızın içinde yayılımları(varyansları) minimize et demek.



f,ax = plt.subplots(figsize = (20,10))    

dendrogram(merg,leaf_rotation = 90) #merg benim scipy kütüphanesinin hieararcical algoritmam olmus oluyo. 

                                    #leaf_roration ise x ekseni değerlerini eksene 90 derece gelecek sekilde yazması.

#burda dendogramın üzerine cizgi çirdirmeye çalıstım ama tam olmadı

#burda data point noktaları 2 boyıtlu alıp kendisi ona göre x boyutunda bir değer atıyor.

a = np.linspace(800,800,3000) #600den basla 600e kadar,3000 tane x değeri var ondan

b = np.arange(1,3001,1)

plt.scatter(b,a,color = "black")    

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
#HC

from sklearn.cluster import AgglomerativeClustering

hieartical_clustering = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")

#burdaki n_cluster seviyesini dendogramdaki trashol a göre belirliyoruz.

#affinity ise öklid distance a göre sınıflandırma yap. linkage ise metodumuz ward (minimiza varyans)

cluster = hieartical_clustering.fit_predict(df)

cluster
df["label"] = cluster
plt.scatter(df.x[df.label==0],df.y[df.label==0],color = "red")  #x ve y eksenleri label ları 0 olanlar

plt.scatter(df.x[df.label==1],df.y[df.label==1],color = "blue") #x ve y eksenleri label ları 1 olanlar

plt.scatter(df.x[df.label==2],df.y[df.label==2],color = "green")#x ve y eksenleri label ları 2 olanlar



plt.show()
a = np.linspace(10,20,5)  #10 ile 20 arasına 5 eşit sayı

a

b = np.arange(1,10,2) #1den basla 10 a kadar 2ser artırarak dizi olustur.

b
#burda data olarak bir twitter datasetini kullancaz. Twitterdan data üretmek için Twitter API konusuna ait !!!

df = pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding = "latin1") 

#datasetinde latin harfleri oldugundan dolayı. r ise read demek

df.head()
#ben burda datamda cinsiyetler ve atılan yorumlar var. Amac atılan yorumlara göre atan kişinin cinsiyetini tahmin etme

#ondan sadece gender ve yorumları alıp yeni bir data olusturuyorum.

df = pd.concat([df.gender,df.description],axis = 1)

df.head(10)
df.info()

#nan valuelar var görüldüğü gibi
df.dropna(axis = 0 , inplace = True) #satır olarak nan value ları at
df.info()
#gender lar burda string halde. kadınları 1 erkekleri 0 yapalım

df.gender = [1 if i =="female" else 0 for i in df.gender]

df.head()
#cleaning data

#burda datada sacma sapan karakterler gülücükler var. Bunlar için Regular Expression yapacaz. Bunun için bir kütüphane var

import re

first_description = df.description[4]

first_description

#burda görüldüğü gibi :) karakterleri var bunları atacağız.re kütüphanesi ile
description = re.sub("[^a-zA-Z]"," ",first_description) 

#diyoruz ki [^a-zA-Z] bu adan zye olan büyük veya küçük harfleri secme(^). Secmediklerini boşluk ile değiştir.

description
description2 = re.sub("[a-zA-Z]"," ",first_description)

description2 #görüldüğü gibi ^ işareti seçme demek. koymayınca seciyo ve harfleri değiştiriyor.
#bütün harfleri küçük yapma

description = description.lower()

description
#stopwords (irrelevant words) gereksiz kelimeler mesela bunlar and,the,a gibi kelimeler bunlar grammerle alakalı kadın erkek ayrımı ile alakası yok.

import nltk

#nltk.download("stopwords") #kaggle dısında bunu bilgisayara indirmek için bu kullanılmalı

from nltk.corpus import stopwords
#stopwords leri import ettik. Benim bu yorumları önce bir tek tek kelime kelime ayırmam lazım ki stopwordler ile karsılastırayım ve onları atım

description = description.split() #default u zaten boşluğa göre

description
#split yerine tokenize methodunu kullanabiliriz bunun bir artısı da var örneğin 'shouldn't ve guzel' stringini split ile 3e ayrılacaktır.

#ama tokenize ile should n't ayrı alıp 4 kelime yapacaktır!!!

description = df.description[4]

description = re.sub("[^a-zA-Z]"," ",description)

description = description.lower()



description = nltk.word_tokenize(description)

description
#stopwords leri çıkarma

description = [i for i in description if not i in set(stopwords.words("english"))]

#stopwors un içindeki english kelimeleri çünkü içinde bir sürü dilin kelimelri var. set ise unique ifadeler için.yani hızlı olsun diye
description
#Lemmazation= bu şu demek mesela maça gitmek çok güzel,maç iyiydi,maçını seveyim gibi burda maç kelimesinin çekimi var

#bunun için bilgisayarda aslında bunlar hepsi ayrı be kelimedir. ama önemli olan asıl kökü olan maç kelimesidir.

#Köküne inmeyi öğreneceğiz

import nltk

lemma = nltk.WordNetLemmatizer()

description = [ lemma.lemmatize(i) for i in description]

description
#join methodu:gerekli işlemlerden sonra artık tekrar metin haline cevirelim listemizi

description = " ".join(description)

description
#Şimdi bütün data üzerine bu işlemleri ayrı ayrı uygulayacaz

#Sırasıyla harf olmayanları temizleme,küçük harf,stopwordsleri temizleme,kelimenin köküne inme

description_list = list()

for i in df.description:

    i = re.sub("[^a-zA-Z]"," ",i) #a'dan z'ye olmayanları seç ve yerine boşluk koy

    i = i.lower()                #küçük harf

    i = nltk.word_tokenize(i)    #metni kelime kelime ayır liste yap

    i = [word for word in i if not word in set(stopwords.words("english"))]  #listedeki kelimelri stopword ile karşılastır

    

    lemma = nltk.WordNetLemmatizer()

    i = [lemma.lemmatize(word) for word in i]   #kelimenin köküne in

    

    i = " ".join(i)   #listeyi boşluk ile birleştir 

   

    description_list.append(i)

    

#cok zaman alırsa stopwords yüzündendir onu kaldırırsın ilerde baska yolla o problemi çözcez.
description_list
#bag of words !!! temel su. her ayrı unique kelime bir feature(sütun) oluyo ve cümleler satır oluyo. varsa cümlede 1 yoksa 0 oluyo

#Bunun için sklearn kütüphanesinde method var

from sklearn.feature_extraction.text import CountVectorizer

maxi = 500 #burda olay su 16 bin tane cümle var.E her cümlede 2 unique kelime olsa 32bin feature eder ve baya uzun sürer

                  #diyoruz ki max feature mız en cok kullanılan 500 kelime olsun

count_vector = CountVectorizer(max_features = maxi,stop_words = "english") #burda stop_words u yapabiliyoduk. Hani yukarıda uzun sürerse

                                                                           #burda küçük harfi lowercase= parametresi ile

                                                                           #gereksiz karakteri de tokken_pattern= parametresi ile yapabilirdik.

#bu yazdıgım parametreler ile bi denersin

sparce_matrix = count_vector.fit_transform(description_list)

sparce_matrix

#bunu isim olarak gösterdi array için toarray methodu
sparce_matrix = count_vector.fit_transform(description_list).toarray()
sparce_matrix
#peki ben bu 500 tane feature ım ne bakmak istersem

count_vector.get_feature_names()
#Text Classification

#burda yaptıgımız gibi datamızı train test split 

#burda x imiz aslında bizim sparce matrix imiz

y = df.iloc[:,0].values.reshape(-1,1) #male or female classes

x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)

#Burda naive baise algoritmasını kullancağız yani tamamen keyfi

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

#Prediction

y_pred = nb.predict(x_test)

print("Accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
from sklearn.feature_extraction.text import CountVectorizer

maxi = 5000 #burda olay su 16 bin tane cümle var.E her cümlede 2 unique kelime olsa 32bin feature eder ve baya uzun sürer

                  #diyoruz ki max feature mız en cok kullanılan 500 kelime olsun

count_vector = CountVectorizer(max_features = maxi,stop_words = "english") #burda stop_words u yapabiliyoduk. Hani yukarıda uzun sürerse

                                                                           #burda küçük harfi lowercase= parametresi ile

                                                                           #gereksiz karakteri de tokken_pattern= parametresi ile yapabilirdik.

#bu yazdıgım parametreler ile bi denersin

sparce_matrix = count_vector.fit_transform(description_list)



sparce_matrix = count_vector.fit_transform(description_list).toarray()



y = df.iloc[:,0].values.reshape(-1,1) #male or female classes

x = sparce_matrix



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)





from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)



y_pred = nb.predict(x_test)

print("Accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
#Bu sefer datasetini sklearn den iris dataseti

from sklearn.datasets import load_iris

iris = load_iris()

iris
#dataframe çevirecem. Bu iris arrayin içinde yazıyolar aynı dataframe de featureları . ile yazıyorduk ya

data = iris.data

feature_names = iris.feature_names

y = iris.target #çiçeklerin türleri 3 tane vardı bunu 0 1 ve 2 diye yazdılar aynı sekilde target names leri de var



df = pd.DataFrame(data,columns=feature_names)

df
df["labels"] = y
x = data

type(x)
#amacım benim datayı görselleştirmek.Ama 4 boyut var bunu söyle yapabilirdik. 3 boyutlu grafik artı bi de renk kullanırdım

#ama ben 2 boyutluya indirip öyle yapacam

from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True) #n_components=2 demek orjinal datamı 2featurea 2 boyuta indir.

                                           #whiten ise normalization

pca.fit(x) #normalde x,y yapıyorduk.ama burda bir test eğitim yapmıyoruz.ondan y ile sınıflarla işimiz yok. amac 2boyutlu görselleştirme

x_pca = pca.transform(x) #burda sadece fit ile modeli kuruyosun. Datama uygulayabilmek için transform demem gerekir.



x_pca
#burda hangisini principal component hangisinin second component için variance bakcaz

print("Variance Ratio: ",pca.explained_variance_ratio_)

#yüzde 92 cok daha büyük oldugundan principal 1.si
#Peki ben 4ten 2ye düşürdüm bu varyansları. Ama ne nekadarlık datamı koruyabildim

print("Sum: ",sum(pca.explained_variance_ratio_))

#yüzde 97lik varyansa yani datamın yüzde 97sine hala sahibim.Yüzde 3lük bilgi kaybı yasamısım
#2D görselleştirme

x_pca.shape
#bu componentleri dataframe ekliyorum.

df["p1"] = x_pca[:,0]

df["p2"] = x_pca[:,1]

df

colors = ["red","green","blue"]

plt.figure(figsize = (15,10))

for i in range(3):  #çünkü 3 türüm var. 0,1,2

    plt.scatter(df.p1[df.labels == i],df.p2[df.labels==i],color=colors[i],label=iris.target_names[i])

plt.xlabel("p1")

plt.ylabel("p2")

plt.legend()

plt.show()
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data

y = iris.target
#KNN yapcaz önce normalization

x = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) #K=3
#K FOLD CV, K=10 sececem. genelde literatürde 10 secilir

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X= x_train, y=y_train, cv = 10) #estimator=hangi algoritmayı kullancan, cv = K fold un K değeri

accuracies
#avarege accuracy

print("Average accuracy: ",np.mean(accuracies))

print("Standard Deviation: ",np.std(accuracies)) 

#tutarlı bir data
dir(np)
#daha sonra artık ferçek test asaması, knn deki k değerine karar verdikten sonra 

#Test

#Ama ondan önce biz crosfold da knn i fit etmiş olsak da burda fit etmemiz gerek

knn.fit(x_train,y_train)

print("Test Accuracy: ",knn.score(x_test,y_test))
#iris dataset for grid search with knn

from sklearn.model_selection import GridSearchCV



grid = {"n_neighbors":np.arange(1,50)} #knn deki k değerlerim 1den 50 ye kadar olsun

knn = KNeighborsClassifier()



grid_knn = GridSearchCV(knn,grid,cv=10) #knn algoritması,knn değerleri,cv için K değeri

grid_knn.fit(x,y)
#print hyperparamater(knn deki k değeri)

print("tuned hyperparameters K: ",grid_knn.best_params_) #tuned ayarlanmıs demek.en iyi k değeri

print("En iyi K değerinin accuracy(best score): ",grid_knn.best_score_)
#grid search with logistic regression(output binary!!)

#bundan dolayı datasetim iris 3tü 2ye düşüyorum. İlk 100 satır 2ye kadar olan ciçek türleri içindi

from sklearn.linear_model import LogisticRegression

x = x[:100,:]

y = y[:100]
grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #bu regulazation ama anlatmadı agır olabilir diye

#l1:lasso l2:ridge.. yani aslında bunlar benim logistic regressionımı hyperparameters

logreg  = LogisticRegression()

grid_lr = GridSearchCV(logreg,grid,cv=10)

grid_lr.fit(x,y)



print("tuned hyperparameters K: ",grid_lr.best_params_) #

print("En iyi hyperparametrelerine göre accuracy(best score): ",grid_lr.best_score_)