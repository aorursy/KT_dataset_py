import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
veri = "../input/my-data/3232/"
# İçerisinde modeli eğitmek için kullanacağımız veriyi içeren klasörün konumu

sınıf_listesi = '../input/labels/labels.csv'
# Kare, üçgen, daire ve yıldız şeklinde ayrılmış olan sınıfları içeren csv dosyası

batch_miktarı = 50
# Her forward ve back propagation işleminde tek seferde 50 resim kullanmayı seçiyoruz.

adım_başı_epoch = 2000
epochs = 3

# Elimizdeki verileri seçtiğimiz batch ve adım_başı_epoch değerleri miktarında eğitiyoruz.
# Eğitimi her bir sefer tamamladığımızda bu işlem bir epoch etmiş oluyor.
# Modelimizi daha iyi eğitmek istersek epoch değerini yükseltebilir ve daha kesin sonuçlar elde edebiliriz.

resimlerin_ebatı = (32, 32, 3)
# Modeli eğitmek için kullanılan verilerin boyutları.

test_oranı = 0.2
# Bulunan verilerin %20'lik kısmı test için ayrılıyor.

doğrulama_oranı = 0.2
# Aynı şekilde modelin cevaplarının doğruluğunun tespiti için de verinin bir kısmının ayrılması gerekiyor.
# %20'si test için ayrılmış olan büyük verinin kalan kısmının %20'si de bu kısım için ayrılır.
#Kütüphane
sayı = 0
resimler = []
klasör_numarası = []
listem = os.listdir(veri)
# os.listdir ile listem değişkeni içerisine veri olarak adlandırdığımız dosya yolu aktarılıyor.
print("Toplam klasör sayısı:", len(listem))
# len(listem) ile listem değişkenin sahip olduğu sayı değerini elde edip print ile yazdırıyoruz.
klasör_sayısı=len(listem)
# Aynı şekilde bu sayı değeri klasör_sayısı değişkenine atanıyor.
print("Klasörler sınıflanmış olarak sırayla import ediliyor...")
for x in range (0,len(listem)):
# for döngüsü kullanılarak range() ile sıfırdan sahip olunan klasör sayısına kadar olan sayı dizini x değişkeni
# içerisine aktarılıyor.

    resim_listesi = os.listdir(veri+"/"+str(sayı))
    # os.listdir ile train4 klasörü içerisindeki 0, 1, 2 ve 3 klasörlerinin içerisine sırayla ulaşıyoruz.
    for y in resim_listesi:
        taranan_resimler = cv2.imread(veri+"/"+str(sayı)+"/"+y)
        # cv2 kütüphanesi ile .imread komutunu kullanılarak içerisinde bulunulan klasördeki resimler bilgisayar tarafından
        # sırayla okunuyor.
        resimler.append(taranan_resimler)
        # daha sonra bu okunan resimlerin her biri başta içerisi boş olan 'resimler' içerisine append ile ekleniyor.
        klasör_numarası.append(sayı)
        # klasör numarası içine de döngü başına 1'er 1'er artan 'sayı' değişkeni aktarılıyor.
    print(sayı, end =" ")
    # end=" " ile normalde alt alta yazılacak olan klasör isimleri yan yana olacak şekilde terminal ekranına yansıtılıyor.
    sayı +=1
    # 0, 1, 2 ve 3 isimli klasörlere sırayla geçiş yapılma olayı sayının her döngüde 1 arttırılması ile gerçekleşiyor.
print(" ")

resimler = np.array(resimler)
# değişkeni matris haline getirmek için np.array komutunu kullanıyoruz.
klasör_numarası = np.array(klasör_numarası)
# değişkeni matris haline getirmek için np.array komutunu kullanıyoruz.

X_train, X_test, y_train, y_test = train_test_split(resimler, klasör_numarası, test_size=test_oranı)
# Yukarıda test ve doğrulama için belirlenilen oranlarda verinin bölünme işlemi burada gerçekleşiyor.
# X_train ile modelimizi oluştururken validation ile eğitilen modelin doğru eğitilip eğitilmediğini görüyoruz.
# daha sonra doğru eğitilmiş modeli gerçek hayat durumlarında test edebilmek için test kısmını kullanıyoruz.
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=doğrulama_oranı)
# test_size kısmında verinin hangi oranda bölüneceği belirtiliyor.





print("Test ve doğrulama için ayrılmış verinin bilgileri...")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)

assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(resimlerin_ebatı))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(resimlerin_ebatı))," The dimesionas of the Validation images are wrong "
assert(X_test.shape[1:]==(resimlerin_ebatı))," The dimesionas of the Test images are wrong"
data=pd.read_csv(sınıf_listesi)
# Pandas kütüphanesi yardımı ile pd.read_csv sayesinde csv dosyası okutuluyor.
print("data shape ",data.shape,type(data))
# Sonra da data değişkeni içine atanan csv verisinin bigileri yazdırılıyor.

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img) # Resimlerin sahip olduğu renkleri cv2 ile siyah beyaz hale getirmek için
    img = equalize(img)  # Bütün resimlerin parlaklığını standarda oturtmak için
    img = img/255        # Neural Network daha hızlı çalışsın ve resimler normalize edilsin diye 255'e bölüm gerçekleşir
    return img



X_train=np.array(list(map(preprocessing,X_train)))
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
# İlk kısımlarda olduğu gibi np.array 'i bu sefer işlem görmüş resimler üzerinde X_train, X_validation ve X_test için
# kullanılıyor.

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

# Kerasta eğitim gerçekleştirmek için verilerin 3D matris hale getirilmesi gerekir.
# Yani resimlerin tamamının boyutunu cv2 ile 32x32 hale getirdikten sonra eğitimin gerçekleşmesi için  bir de
# 32x32x1 şeklinde 3D matris haline getirmek zorundayız.
dataGen= ImageDataGenerator(width_shift_range=0.1,   # genişliği belirtilen yüzdede sağa 								sola kaydırmak için
                            height_shift_range=0.1,  # yüksekliği belirtilen yüzdede sağa 								sola kaydırmak için
                            zoom_range=0.2,          # zoom yapılması için
                            shear_range=0.1,         # kırpma yapılması için
                            rotation_range=10)       # resmin döndürüleceği değer için

# Bu kısımda eğitimde overfitting probleminin çözülmesi için elimizde varolan resimler üzerinde küçük oynamalar yaparak
# onları tekrar oluşturup eğitimde kullanıyoruz. Bunun sayesinde hem sahip olunan veri zenginleşiyor hem de bilgisayarın
# öğrendiği şeyi farklı açı ve boyutlarda görmesi durumunda bile sonuç daha iyi hale geliyor.

dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)
X_batch,y_batch = next(batches)
# dataGen değişkeninde gerçekleşen işlemleri .fit komutu ile X_train'e uyguluyoruz.

y_train = to_categorical(y_train, klasör_sayısı)
y_validation = to_categorical(y_validation,klasör_sayısı)
y_test = to_categorical(y_test,klasör_sayısı)
def myModel():
    filtre_sayısı = 60
    filtre_boyutu = (5, 5)
    filtre_boyutu_2 = (3, 3)
    pool_boyutu=(2, 2)  # overfitting(ezberleme) miktarını azaltmak için
    düğüm_sayısı = 500  # Gizli katmandaki düğüm sayısı
    model = Sequential()
    # Modelimizi oluştururken Sequential metodunu kullanıyoruz.

    model.add((Conv2D(filtre_sayısı, filtre_boyutu, input_shape=(resimlerin_ebatı[0], resimlerin_ebatı[1], 1), activation = 'relu')))
    # Genel olarak ne kadar filtre eklersek o kadar doğru sonuçlar elde edeceğimizden tek filtre yerine birkaç filtre
    # birden kullanıyoruz.

    model.add((Conv2D(filtre_sayısı, filtre_boyutu, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_boyutu))
    # overfitting(ezberleme) miktarını azaltmak için MaxPooling işlemi gerçekleştiriliyor.

    model.add((Conv2D(filtre_sayısı//2, filtre_boyutu_2, activation='relu')))
    # relu aktivasyon fonksiyonunu kullanarak karmaşıklığı arttırıyoruz.
    model.add((Conv2D(filtre_sayısı // 2, filtre_boyutu_2, activation='relu')))
    model.add(MaxPooling2D(pool_size=pool_boyutu))
    model.add(Dropout(0.5))
    # forward propogation yaparken rastgele nöronlar seçiliyor ve devre dışı bırakılıyor. Bu metoda Dropout metodu denir
    # Devre dışı bırakılacak nöronlar belirlenmiş threshold değerinin altında değere sahip nöronlardır.
    # Bu metodun kullanımı da overfitting yani ezberleme olayını azaltma konusunda yardımcı bir rol oynar.

    model.add(Flatten())
    # Filtreleme ve eleme işlemlerinin tamamlanması sonrasından artık artificial neural network kullanacağımız için
    # Flatten işlemi gerçekleştirilmelidir. Artificial neural network içerisinde matrisi kullanabilmek için
    # mesela 3,3 lük matrisi 9,1 'lik şekilde Flatten ile düzleştirmeliyiz.
    model.add(Dense(düğüm_sayısı, activation='relu'))
    # Bu kısımda Gizli Katman ekleniyor.
    model.add(Dropout(0.5))
    model.add(Dense(klasör_sayısı, activation='softmax'))  # Çıkış Filtresi
    # Çoklu sınıf ayrımı yaptığımız için sigmoid fonksiyonunun daha genelleştirilmiş bir hali olan softmax aktivasyon
    # fonksiyonunu kullanarak çıkış filtresini ekliyoruz.

    # Modelin Compile Edilmesi
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Adam = adaptive momentum optimizer , lr = learning rate
    # Optimizer kullanımının sebebi eğitim gerçekleşirken en iyi performansın sağlanması adına learning rate'in
    # ayarlanması yani gereken miktarda arttırılması.
    # loss fonksiyonu ise sinir ağının cevabının hata oranı azalana dek weight değerlerinin güncellenmesini sağlar.
    return model
model = myModel()
print(model.summary())
# Model hakkındaki bilgilerin terminale yansıtılması işlemi
history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batch_miktarı), steps_per_epoch=adım_başı_epoch, epochs=epochs, validation_data=(X_validation, y_validation), shuffle=1)
# Keras'ın .fit_generator komutu ile hazırlığı tamamlanmış modelin eğitimi başlatılıyor.

plt.figure(1)
# İlk grafik
plt.plot(history.history['loss'])
# Eğitilmiş modelimizin değerlerini grafikteki değerlere aktarıyoruz ki grafikte gösterebilelim
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
# Grafiğin başlığı
plt.xlabel('epoch')
# x ekseninde yazılacak olan kısım yani 'epoch'
plt.figure(2)
# 2. Grafik
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
# Grafiğin köşe kısmında training ve validation yazılı küçük bir tablo oluşmasını sağlar.
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
# Grafiği oluşturmamızı sağlayan matplotlib kütüphanesinin yine farklı bir komutu olan plt.show() ile hazırlanmış olan grafikler sırayla ekranda gösteriliyor.
score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
pickle_out= open("eniyisi.p", "wb") 
pickle.dump(model,pickle_out)
pickle_out.close()
cv2.waitKey(0)

# Son olarak eğitimi tamamlanmış olan modeli bir pickle objesine kaydediyoruz.
# Bunun sebebi bu modeli daha sonra gerçek zamanlı olarak opencv fonksiyonu içerisinde kullanacak olmamız