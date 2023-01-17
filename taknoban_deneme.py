# veri çeşitlendirme

from keras.preprocessing.image import ImageDataGenerator



from keras.preprocessing import image



# eniyileme

from keras import optimizers



# CNN katmanları

from keras.layers import Conv2D



# Optimizer

from keras.optimizers import Adam



# model tanımlamak

from keras.models import Sequential,Model



# enbüyükleri biriktirme

from keras.layers import MaxPooling2D, GlobalAveragePooling2D



# çıktı boyutunu düzenleme

from keras.layers import Flatten



# tamamen bağlı katman

from keras.layers import Dense



# iletim sönümü

from keras.layers import Dropout



# kayıpları ve başarımı görselleştirme

import matplotlib.pyplot as plt



# matris işlemleri

import numpy as np



# klasör ve işletim sistemi

import os
train_path = "../input/skin-cancer-malignant-vs-benign/train/"

test_path = "../input/skin-cancer-malignant-vs-benign/test/"
# eğitim verisinden geçiş sayısı (tur)

epochs = 20

#aynı anda işlenen görüntü sayısı

batch_size = 5  
train_datagen = ImageDataGenerator(



      # resim pixellerini 0,1 arasına sıkıştırma

      rescale=1./255,



      # derece cinsinden (0-180) resimlerin rastgele döndürülme açısı

      rotation_range=40,



      # resimlerin yatayda ve dikeyde kaydırılma oranları

      width_shift_range=0.2,



      # resimlerin yatayda ve dikeyde kaydırılma oranları

      height_shift_range=0.2,



      # burkma işlemi

      shear_range=0.2,



      # yakınlaştırma işlemi

      zoom_range=0.2,



      # dikeyde resim döndürme

      horizontal_flip=True,



      # işlemlerden sonra ortaya çıkan  fazla 

      # görüntü noktalarının nasıl doldurulacağını belirler

      fill_mode='nearest')





# test resimlerinde çeşitlendirme yapmıyoruz.

test_datagen = ImageDataGenerator(rescale=1./255)



# çeşitlendirilmiş verileri kullanmak (eğitim)

train_generator = train_datagen.flow_from_directory(



        # hedef dizin

        train_path,



        # tüm resimler (150x150) olarak boyutlandırılacak

        target_size=(150, 150),



        # yığın boyutu

        batch_size=20,



        # binary_crossentropy kullandığımız için

        # ikili etiketler gerekiyor.

        class_mode='binary')



# verileri kullanmak (doğrulama)

validation_generator = test_datagen.flow_from_directory(



        test_path,



        target_size=(150, 150),



        batch_size=20,



        class_mode='binary')
from keras.applications import VGG16

deneme_model= VGG16()



print(deneme_model.summary())
temel_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
ust_model = Sequential()

ust_model.add(GlobalAveragePooling2D(input_shape=temel_model.output_shape[1:], data_format=None)),  

ust_model.add(Dense(256, activation='relu'))

ust_model.add(Dropout(0.5))

ust_model.add(Dense(1, activation='sigmoid')) 
model = Model(inputs=temel_model.input, outputs=ust_model(temel_model.output))
model.compile(loss='binary_crossentropy', # kayıp fonksiyonu



              # eniyileme:

              # ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde

              # bulundurarak kendisini güncelleme mekanizması

              optimizer=Adam(lr = 0.0001),

              

              # eğitim ve test süresince takip edilecek metrikler.

              metrics=['acc'])
nb_train_samples = 500

nb_validation_samples = 10



history = model.fit_generator(

            train_generator,

            steps_per_epoch=nb_train_samples // batch_size,

            epochs=epochs,

            validation_data=validation_generator,

            validation_steps=nb_validation_samples // batch_size)
# --- sonuçları görselleştirme --- #





# Eğitim başarım skoru

acc = history.history["acc"]



# doğrulama başarım skoru

val_acc = history.history["val_acc"]



# eğitim kayıp skoru

loss = history.history["loss"]



# doğrulama kayıp skoru

val_loss = history.history["val_loss"]



# epochs sayısına göre grafik çizdireceğiz.

epochs = range(1, len(acc) + 1)



# eğitim başarımını kendine özel çizdirdik.

plt.plot(epochs, acc, "bo", label="Eğitim başarımı")



# doğrulama başarımını kendine özel çizdirdik.

plt.plot(epochs, val_acc, "r", label="Doğrulama başarımı")



# çizdirmemizin başlığı

plt.title("Eğitim ve doğrulama başarımı")



plt.legend()



plt.figure()



# eğitim kaybını kendine özel çizdirdik.

plt.plot(epochs, loss, "bo", label="Eğitim kaybı")



# doğrulama kaybını kendine özel çizdirdik.

plt.plot(epochs, val_loss, "r", label="Doğrulama kaybı")





# çizdirmemizin başlığı

plt.title("Eğitim ve doğrulama kaybı")



plt.legend()



# ekrana çıkartma

plt.show()