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
train = pd.read_csv("../input/mitbih_train.csv", header=None)

test = pd.read_csv("../input/mitbih_test.csv",header=None)

print(train.info())

print("**********************************************")

print(train.describe())
train[187].value_counts() # 187.sütun label'larımızın yani class larımızın olduğu sınıftı

train.tail(3)
#print(train.loc[0,:])

x = np.arange(0, 187)*8/1000 #ms cinsinden x ekseni çizdirmek için 

x.shape

print(train.loc[0,0:186].shape)



#labellarımızı int' e çevirelim daha güzel olur

# train.dtypes

train[187]=train[187].astype(int)

train.dtypes



C0=train[train[187] == 0] #CAT N

C1=train[train[187] == 1] #CAT S

C2=train[train[187] == 2] #CAT V

C3=train[train[187] == 3] #CAT F

C4=train[train[187] == 4] #CAT Q



print(str(len(C0)) + " : Cat N train örnek sayisi")

print(str(len(C1)) + " : Cat S train örnek sayisi")

print(str(len(C2)) + " : Cat V train örnek sayisi")

print(str(len(C3)) + " : Cat F train örnek sayisi")

print(str(len(C4)) + " : Cat Q train örnek sayisi")
#loc ile iloc kullanımına bak

x = np.arange(0, 187)*8/1000 #ms cinsinden x ekseni çizdirmek için 

plt.figure(figsize=(20,12))

plt.plot(x, train.loc[0,0:186], label="Cat. N")

plt.plot(x, train.loc[1,0:186], label="Cat. N")

plt.plot(x, train.loc[2,0:186], label="Cat. N")

plt.plot(x, train.loc[3,0:186], label="Cat. N")

plt.plot(x, train.loc[4,0:186], label="Cat. N")

plt.plot(x, train.loc[5,0:186], label="Cat. N")

plt.plot(x, train.loc[6,0:186], label="Cat. N")

plt.plot(x, train.loc[7,0:186], label="Cat. N")

plt.plot(x, train.loc[8,0:186], label="Cat. N")

plt.legend()

plt.title("9-beat ECG CAT N category", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.show()

C4.head(3)
#loc ile iloc kullanımına bak

x = np.arange(0, 187)*8/1000 #ms cinsinden x ekseni çizdirmek için 

plt.figure(figsize=(20,12))

plt.plot(x, C0.iloc[9,0:187], label="Cat. N")

plt.plot(x, C1.iloc[12,0:187], label="Cat. S")#indexleri resetlemeden kopardik ondan böyle alıcaz

plt.plot(x, C2.iloc[4,0:187], label="Cat. V")

plt.plot(x, C3.iloc[123,0:187], label="Cat. F")

plt.plot(x, C4.iloc[23,0:187], label="Cat. Q")

plt.grid()



plt.legend()

plt.title("1-beat(random) ECG for every category", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.show()

"""

#CAT N

C1=train[train[187] == 1] #CAT S

C2=train[train[187] == 2] #CAT V

C3=train[train[187] == 3] #CAT F

C4=train[train[187] == 4] #CAT Q"""



"""

plt.figure(figsize=(20,12))

plt.plot(x, X[C0, :][0], label="Cat. N")

plt.plot(x, X[C1, :][0], label="Cat. S")

plt.plot(x, X[C2, :][0], label="Cat. V")

plt.plot(x, X[C3, :][0], label="Cat. F")

plt.plot(x, X[C4, :][0], label="Cat. Q")

plt.legend()

plt.title("1-beat ECG for every category", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.show()

"""

#loc ile iloc kullanımına bak

x = np.arange(0, 187)*8/1000 #ms cinsinden x ekseni çizdirmek için 



plt.figure(figsize=(15,12))

# örnek degerler için 9,12,4,123,23 hocaya danış

plt.subplot(3, 2, 1)

plt.plot(x, C0.iloc[2000,0:187], '.-',color="blue")

plt.title('A tale of 5 subplots')

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 2)

plt.plot(x, C1.iloc[12,0:187], '.-',color="orange",)

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.subplot(3, 2, 3)

plt.plot(x, C2.iloc[4,0:187], '.-',color="green")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 4)

plt.plot(x, C3.iloc[123,0:187], '.-',color="red")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 5)

plt.plot(x, C4.iloc[23,0:187], '.-',color="purple")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.show()
print(len(C0))

print(len(C1))

print(len(C2))

print(len(C3))

print(len(C4))

#bitirme ek 12:34 sinyali devam ettirmemiz gerek

print(C3.shape)

#labellarımızı silelim zaten en sonda biz ekliyoruz tekrar sorun yok

C3=C3.drop([187], axis=1)

C0=C0.drop([187], axis=1)

C1=C1.drop([187], axis=1)

C2=C2.drop([187], axis=1)

C4=C4.drop([187], axis=1)



print(C3.shape)
C3=pd.concat((C3,C3), axis=1)

C0=pd.concat((C0,C0), axis=1)

C1=pd.concat((C1,C1), axis=1)

C2=pd.concat((C2,C2), axis=1)

C4=pd.concat((C4,C4), axis=1)



print(C3.shape)

C3.head(2)
C3_new = C3.values #c3 dataframe ini array'e çevirdik

len(C3_new[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

step_size = 100 #bir örnekteki step size sütun cinsinden         

oteleme = 20#öteleme miktarı sütun cinsinden



train_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C3_new.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        train_x.append(C3_new[k,i:i+step_size])

    

    
#loc ile iloc kullanımına bak

x = np.arange(0, 374)*8/1000 #ms cinsinden x ekseni çizdirmek için 

plt.figure(figsize=(20,12))

plt.plot(x, C3_new[600,0:374], label="Cat. N")



plt.legend()

plt.title("9-beat ECG CAT N category", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.show()
print(C3_new.shape)

print("C3_new type : ",type(C3_new))

print(np.shape(train_x)) #çerçeveleme işlemi tamam 

print("train_x type : ",type(train_x))

print("np.asarray(a) komutu ile 'train x' tipi:", type(np.asarray(train_x)))

train_x_C3=np.asarray(train_x) # listemizi numpy array' e çevirdik

print("train_x_C3 shape :",train_x_C3.shape)

print("train_x_C3 size : (nxm) : ",train_x_C3.size)

print("sample sayisi : ",len(train_x_C3))

print("bir sample daki feature sayisi : ", len(train_x_C3[0]))

#şimdi buna karşılık gelen bir train_x_C3 labelları olmak zorunda biz 5 class yerine 2 class kullanalım diye 0'lardan oluşan

#bir matris elde etmemiz gerekir .

print(np.zeros(len(train_x_C3)))

train_y_C3=np.zeros(len(train_x_C3),dtype=int)

print("C3 kendimizin oluşturduğu trainy : ",train_y_C3.shape)

train_x_C3.shape
x = np.arange(0, 100)*8/1000 #ms cinsinden x ekseni çizdirmek için 



plt.figure(figsize=(15,12))

# örnek degerler için 9,12,4,123,23 hocaya danış

plt.subplot(3, 2, 1)

plt.plot(x, train_x_C3[1024,:], '.-',color="blue")

plt.xlabel('time (ms)')

plt.ylabel('Amplitude')

plt.grid()

"şimdi diğer class larımızıda c3 formatında yeni dizilerimize sokmak için işlemleri yapalım "

"bu sefer biraz daha seri olucaz cünkü mantalite aynı :) "

C4_new = C4.values #c4 dataframe ini array'e çevirdik

train_x = []

for k in range(0,C4_new.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists

        train_x.append(C4_new[k,i:i+step_size])

        

train_x_C4=np.asarray(train_x) # listemizi numpy array' e çevirdik



print(C4_new.shape)

print("C4_new type : ",type(C4_new))

print(np.shape(train_x)) #çerçeveleme işlemi tamam 

print("train_x type : ",type(train_x))

print("np.asarray(a) komutu ile 'train x' tipi:", type(np.asarray(train_x)))



#şimdi buna karşılık gelen bir train_x_C4 labelları olmak zorunda 

train_y_C4=np.zeros(len(train_x_C4),dtype=int)

print("C4 kendimizin oluşturduğu trainy : ",train_y_C4.shape)





C2_new = C2.values #c4 dataframe ini array'e çevirdik

train_x = []

for k in range(0,C2_new.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists

        train_x.append(C2_new[k,i:i+step_size])

        

train_x_C2=np.asarray(train_x) # listemizi numpy array' e çevirdik



print(C2_new.shape)

print("C2_new type : ",type(C2_new))

print(np.shape(train_x)) #çerçeveleme işlemi tamam 

print("train_x type : ",type(train_x))

print("np.asarray(a) komutu ile 'train x' tipi:", type(np.asarray(train_x)))



#şimdi buna karşılık gelen bir train_x_C4 labelları olmak zorunda 

train_y_C2=np.zeros(len(train_x_C2),dtype=int)

print("C4 kendimizin oluşturduğu trainy : ",train_y_C2.shape)

C1_new = C1.values #c4 dataframe ini array'e çevirdik

train_x = []

for k in range(0,C1_new.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists

        train_x.append(C1_new[k,i:i+step_size])

        

train_x_C1=np.asarray(train_x) # listemizi numpy array' e çevirdik



print(C1_new.shape)

print("C1_new type : ",type(C1_new))

print(np.shape(train_x)) #çerçeveleme işlemi tamam 

print("train_x type : ",type(train_x))

print("np.asarray(a) komutu ile 'train x' tipi:", type(np.asarray(train_x)))



#şimdi buna karşılık gelen bir train_x_C4 labelları olmak zorunda 

train_y_C1=np.zeros(len(train_x_C1),dtype=int)

print("C1 kendimizin oluşturduğu trainy : ",train_y_C1.shape)
"bura önemli c0 normal beat class'ımız ondan 1 vericez çıkış labellerine"

C0_new = C0.values #c4 dataframe ini array'e çevirdik

train_x = []

for k in range(0,C0_new.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists

        train_x.append(C0_new[k,i:i+step_size])

        

train_x_C0=np.asarray(train_x) # listemizi numpy array' e çevirdik



print(C0_new.shape)

print("C0_new type : ",type(C0_new))

print(np.shape(train_x)) #çerçeveleme işlemi tamam 

print("train_x type : ",type(train_x))

print("np.asarray(a) komutu ile 'train x' tipi:", type(np.asarray(train_x)))



#şimdi buna karşılık gelen bir train_x_C4 labelları olmak zorunda 

train_y_C0=np.ones(len(train_x_C0),dtype=int)

print("C0 kendimizin oluşturduğu trainy : ",train_y_C0.shape)
#loc ile iloc kullanımına bak

x = np.arange(0, 374)*8/1000 #ms cinsinden x ekseni çizdirmek için 

plt.figure(figsize=(20,12))

plt.plot(x, C1_new[1000,0:374], label="Cat. N")



plt.legend()

plt.title("9-beat ECG CAT N category", fontsize=20)

plt.ylabel("Amplitude", fontsize=15)

plt.xlabel("Time (ms)", fontsize=15)

plt.show()
print(" C3 örnek sayısı : ",train_x_C3.shape[0])

print(" C2 örnek sayısı : ",train_x_C2.shape[0])

print(" C1 örnek sayısı : ",train_x_C1.shape[0])

print(" C4 örnek sayısı : ",train_x_C4.shape[0])

print(" C0 örnek sayısı : ",train_x_C0.shape[0])

#şimdi tüm classlar parçalandı bunları tekrar birleştirmemiz gerek eğitimden önce tek bir matris haline getirmemiz gerek

#sırasıyla c0,c1,c2,c3,c4 train matrislerini ve labellarini birleştirelim

#np.hstack((a,b))

Xtrain=np.concatenate((train_x_C0,train_x_C1, train_x_C2,train_x_C3,train_x_C4), axis=0)

Ytrain=np.concatenate((train_y_C0,train_y_C1, train_y_C2,train_y_C3,train_y_C4), axis=0)





print("Xtrain shape = ",Xtrain.shape)

print("Ytrain shape = ",Ytrain.shape)
#loc ile iloc kullanımına bak

x = np.arange(0, 100)*8/1000 #ms cinsinden x ekseni çizdirmek için 



plt.figure(figsize=(15,12))

# örnek degerler için 9,12,4,123,23 hocaya danış

plt.subplot(3, 2, 1)

plt.plot(x, train_x_C0[451,:], '.-',color="blue")

plt.title('A tale of 5 subplots')

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.subplot(3, 2, 2)

plt.plot(x, train_x_C1[451,:], '.-',color="orange",)

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.subplot(3, 2, 3)

plt.plot(x, train_x_C2[451,:], '.-',color="green")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 4)

plt.plot(x, train_x_C3[451,:], '.-',color="red")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 5)

plt.plot(x, train_x_C4[451,:], '.-',color="purple")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.show()
#loc ile iloc kullanımına bak

x = np.arange(0, 100)*8/1000 #ms cinsinden x ekseni çizdirmek için 



plt.figure(figsize=(15,12))

# örnek degerler için 9,12,4,123,23 hocaya danış

plt.subplot(3, 2, 1)

plt.plot(x, train_x_C0[100,:], '.-',color="blue")

plt.title('A tale of 5 subplots')

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.subplot(3, 2, 2)

plt.plot(x, train_x_C0[101,:], '.-',color="orange",)

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.subplot(3, 2, 3)

plt.plot(x, train_x_C0[102,:], '.-',color="green")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 4)

plt.plot(x, train_x_C0[103,:], '.-',color="red")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()



plt.subplot(3, 2, 5)

plt.plot(x, train_x_C0[104,:], '.-',color="purple")

plt.xlabel('time (s)')

plt.ylabel('Amplitude')

plt.grid()





plt.show()
#şimdi test değerlerini biçirlendirelim



test[187].value_counts() # 
#labellarımızı int' e çevirelim daha güzel olur

# test.dtypes

test[187]=test[187].astype(int)

test.dtypes



C0_test=test[test[187] == 0] #CAT N

C1_test=test[test[187] == 1] #CAT S

C2_test=test[test[187] == 2] #CAT V

C3_test=test[test[187] == 3] #CAT F

C4_test=test[test[187] == 4] #CAT Q



print(str(len(C0_test)) + " : Cat N train test sayisi")

print(str(len(C1_test)) + " : Cat S train test sayisi")

print(str(len(C2_test)) + " : Cat V train test sayisi")

print(str(len(C3_test)) + " : Cat F train test sayisi")

print(str(len(C4_test)) + " : Cat Q train test sayisi")
#bitirme ek 12:34 sinyali devam ettirmemiz gerek

print("C3_test shape : ",C3_test.shape)

#labellarımızı silelim zaten en sonda biz ekliyoruz tekrar sorun yok

C0_test=C0_test.drop([187], axis=1)

C1_test=C1_test.drop([187], axis=1)

C2_test=C2_test.drop([187], axis=1)

C3_test=C3_test.drop([187], axis=1)

C4_test=C4_test.drop([187], axis=1)



print("C3_test shape : ",C3_test.shape)

C0_test=pd.concat((C0_test,C0_test), axis=1)

C1_test=pd.concat((C1_test,C1_test), axis=1)

C2_test=pd.concat((C2_test,C2_test), axis=1)

C3_test=pd.concat((C3_test,C3_test), axis=1)

C4_test=pd.concat((C4_test,C4_test), axis=1)



C3_new_test = C3_test.values #c3 dataframe ini array'e çevirdik

len(C3_new_test[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

#step_size = 100 #bir örnekteki step size sütun cinsinden         

#oteleme = 5#öteleme miktarı sütun cinsinden



test_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C3_new_test.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        test_x.append(C3_new_test[k,i:i+step_size])

    
print(C3_new_test.shape)

print("C3_new_test type : ",type(C3_new_test))

print(np.shape(test_x)) #çerçeveleme işlemi tamam 

print("test_x type : ",type(test_x))

print("np.asarray(a) komutu ile 'test_x' tipi:", type(np.asarray(test_x)))
test_x_C3=np.asarray(test_x) # listemizi numpy array' e çevirdik

print("test_x_C3 shape :",test_x_C3.shape)

print("test_x_C3 size : (nxm) : ",test_x_C3.size)

print("sample sayisi : ",len(test_x_C3))

print("bir sample daki feature sayisi : ", len(test_x_C3[0]))

#şimdi buna karşılık gelen bir train_x_C3 labelları olmak zorunda biz 5 class yerine 2 class kullanalım diye 0'lardan oluşan

#bir matris elde etmemiz gerekir .

print(np.zeros(len(test_x_C3)))

test_y_C3=np.zeros(len(test_x_C3),dtype=int)

print("C3 kendimizin oluşturduğu test_y_C3 : ",test_y_C3.shape)

"şimdi train de oldugu için bu adamı her bir test label'i için yapalım.Keşke fonksiyon yazsaydık ya "

#C4 için



C4_new_test = C4_test.values #c3 dataframe ini array'e çevirdik

len(C4_new_test[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

step_size = 100 #bir örnekteki step size sütun cinsinden         

oteleme = 5#öteleme miktarı sütun cinsinden



test_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C4_new_test.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        test_x.append(C4_new_test[k,i:i+step_size])

test_x_C4=np.asarray(test_x)

test_y_C4=np.zeros(len(test_x_C4),dtype=int)

print("test_x_C3 shape :",test_x_C4.shape)

print("Kendimizin oluşturduğu test_y_C3 : ",test_y_C4.shape)
#C1 için



C1_new_test = C1_test.values #c3 dataframe ini array'e çevirdik

len(C1_new_test[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

step_size = 100 #bir örnekteki step size sütun cinsinden         

oteleme = 5#öteleme miktarı sütun cinsinden



test_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C1_new_test.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        test_x.append(C1_new_test[k,i:i+step_size])

test_x_C1=np.asarray(test_x)

test_y_C1=np.zeros(len(test_x_C1),dtype=int)

print("test_x_C1 shape :",test_x_C1.shape)

print("Kendimizin oluşturduğu test_y_C1 : ",test_y_C1.shape)
#C2 için



C2_new_test = C2_test.values #c3 dataframe ini array'e çevirdik

len(C2_new_test[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

step_size = 100 #bir örnekteki step size sütun cinsinden         

oteleme = 5#öteleme miktarı sütun cinsinden



test_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C2_new_test.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        test_x.append(C2_new_test[k,i:i+step_size])

test_x_C2=np.asarray(test_x)

test_y_C2=np.zeros(len(test_x_C2),dtype=int)

print("test_x_C2 shape :",test_x_C2.shape)

print("Kendimizin oluşturduğu test_y_C2 : ",test_y_C2.shape)
#C0 için



C0_new_test = C0_test.values #c3 dataframe ini array'e çevirdik

len(C0_new_test[0])#array uzunlugu



"step_size ve ötelemeyi istedigimiz gibi değiştirebiliriz"

step_size = 100 #bir örnekteki step size sütun cinsinden         

oteleme = 30#öteleme miktarı sütun cinsinden



test_x = []

#burada yapmak istedigim belli bir çerçeve yapısı ve öteleme miktarıyla o satırı dolaşıp sonra bunu yeni data framemimizin

#ilk satırı yapmak böyleyece hem veri sayısı artmıs olucak hemde sinyal belli aralıklarla gerçek zamanlı alınıyormuş gibi olucak

#Burada ilk c3 üstünde deneyeceğiz 

for k in range(0,C0_new_test.shape[0],1):

    

    for i in range(0,186,oteleme): # making feature and the label lists // step_size i silerek sinyalin tamamını almaya calıstık

        test_x.append(C0_new_test[k,i:i+step_size])

test_x_C0=np.asarray(test_x)

test_y_C0=np.ones(len(test_x_C0),dtype=int)

print("test_x_C0 shape :",test_x_C0.shape)

print("Kendimizin oluşturduğu test_y_C0 : ",test_y_C0.shape)



print(" C3 örnek sayısı : ",test_x_C3.shape[0])

print(" C2 örnek sayısı : ",test_x_C2.shape[0])

print(" C1 örnek sayısı : ",test_x_C1.shape[0])

print(" C4 örnek sayısı : ",test_x_C4.shape[0])

print(" C0 örnek sayısı : ",test_x_C0.shape[0])

#şimdi tüm test classlar parçalandı bunları tekrar birleştirmemiz gerek eğitimden önce tek bir matris haline getirmemiz gerek

#sırasıyla c0,c1,c2,c3,c4 train matrislerini ve labellarini birleştirelim

#np.hstack((a,b))

Xtest=np.concatenate((test_x_C0,test_x_C1, test_x_C2,test_x_C3,test_x_C4), axis=0)

Ytest=np.concatenate((test_y_C0,test_y_C1, test_y_C2,test_y_C3,test_y_C4), axis=0)

print("*********************")

print("*********************")

print("Xtest shape = ",Xtest.shape)

print("Ytest shape = ",Ytest.shape)

print("Xtrain shape = ",Xtrain.shape)

print("Ytrain shape = ",Ytrain.shape)
print("Xtest shape = ",Xtest.shape)

print("Ytest shape = ",Ytest.shape)

print("Xtrain shape = ",Xtrain.shape)

print("Ytrain shape = ",Ytrain.shape)
#train ve test verilerimizin yeni hallerini çıktı olarak alalım

#df almak istersek 

#import pandas as pd 

#pd.DataFrame(np_array).to_csv("path/to/file.csv")



Xtest_df=pd.DataFrame(Xtest)

Ytest_df=pd.DataFrame(Ytest)

Xtrain_df=pd.DataFrame(Xtrain)

Ytrain_df=pd.DataFrame(Ytrain)





#denemelik df üzerinden output alalım



np.savetxt("Ytrain.csv", Ytrain, delimiter=",")

np.savetxt("Xtrain.csv", Xtrain, delimiter=",")

np.savetxt("Xtest.csv", Xtest, delimiter=",")

np.savetxt("Ytest.csv", Ytest, delimiter=",")









#numpyarray olarak istersek

#import numpy

#a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])

#numpy.savetxt("foo.csv", a, delimiter=",")
