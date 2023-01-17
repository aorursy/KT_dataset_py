# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

if tf.test.gpu_device_name():

    print('Varsayılan GPU Cihazı : {}'.format(tf.test.gpu_device_name()))

else:

    print("Nvidia GPU Driver, CUDA V10.1 ve cuDNN V7.6.5 yazılımlarını yükleyiniz.")

    os.system("nvidia-smi")

    quit()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Modelimizin görsel halini diske kaydetmek.
from keras.utils import plot_model 

# Xception Öneğitimli Evrişimli Sinir Ağı 
from keras.applications import xception

# veri çeşitlendirme
from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

# eniyileme
from keras import optimizers

# model tanımlamak
from keras.models import Sequential

# çıktı boyutunu düzenleme
from keras.layers import Flatten

# tamamen bağlı katman
from keras.layers import Dense

# iletim sönümü
from keras.layers import Dropout

# kayıpları ve başarımı görselleştirme
import matplotlib.pyplot as plt

from time import sleep as sl

# matris işlemleri
import numpy as np

# klasör ve işletim sistemi
import os, shutil

# keras
import keras

# Yığın normalleştirme
from keras.layers import BatchNormalization

# ----------- veri ön işleme (veri klasör yollarını kodda tanımlama) ----------- #


# Test veri seti klasörü
train_dir = os.path.join("../input/mask-datasets-v1/Mask_Datasets/Train") 


# Doğrulama veri seti klasörü
validation_dir = os.path.join("../input/mask-datasets-v1/Mask_Datasets/Validation")


# Eğitim veri setinde maske takan isnan verileri
train_mask_dir = os.path.join(train_dir,"Mask") 


# Eğitim veri setinde maske takmayan insan verileri
train_no_mask_dir = os.path.join(train_dir,"No_mask")


# Doğrulama veri setinde maske takan isnan verileri
validation_mask_dir = os.path.join(validation_dir,"Mask")


# Doğrulama veri setinde maske takmayan insan verileri
validation_no_mask_dir = os.path.join(validation_dir,"No_mask")



os.system("clear")


# --- Veri seti çeşitlendirme --- #


# veri çeşitlendirme
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
        train_dir,

        # tüm resimler (150x150) olarak boyutlandırılacak
        target_size=(150, 150),

        # yığın boyutu
        batch_size=20,

        # binary_crossentropy kullandığımız için
        # ikili etiketler gerekiyor.
        class_mode='binary')


# verileri kullanmak (doğrulama)
validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')


sl(5)



# --- Xception Öneğitimli Evrişimli Sinir Ağı oluşturma --- #


"""

artık bağlantı Xception da dahil 2015 sonrasında birçok modelde kullanılan

bir diğer çizgisel ağ yapısıdır. 2015 sonlarında ILSVRC ImageNet yarışmasını kazanan

Microsoft 'tan He vd. tarıfından geliştirilmiştir.

Modelleri her büyük çaplı derin öğrenme modelinin başının belası olan iki yaygın problemle mücadele ediyor:

Gradyan yok olması ve gösterimsel darboğaz

"""


conv_base = xception.Xception(weights="imagenet", include_top=False, input_shape=(150,150,3))

print(conv_base.summary())


# Öneğitimli Evrişimli Sinir Ağındaki bazı katmanlar haricinde diğer tüm katmanları donduruyoruz.

# Dondurma sebebimiz ise parametre sayısı çok fazla olunca fazla işlem kapasitesi demektir.

# Dondurma sebebimiz Dense katmanları rastgele başlatıldığından eğitim esnasnında çok büyük güncellemeler 
# alacaktır ve buda daha önce öğrenilen gösterimleri yok edecektir.

# Bu haliyle sadece block14_sepconv1 katmanın ve sonraki katmanların ağırlıkları güncellenecek


conv_base.trainable = True

set_trainable = False

for layer in conv_base.layers:

	if layer.name == "block14_sepconv1":

		set_trainable = True

	if set_trainable:

		layer.trainable = True

	else:

		layer.trainable = False


# model
model = Sequential()

# modelimize Xception Öneğitimli Evrişimli Sinir Ağını ekledik
model.add(conv_base)

# Yığın normalleştirme
# modelin daha iyi geneleştirme yapmasını sağlar.
# eğitim süresince  verinin ortalaması ve standart sapmasının değişimlerine bakarak veriyi normalize eder. 
model.add(BatchNormalization())

# 3B çıktıları 1B vektörlere düzenler
model.add(Flatten())

# tamamen bağlı katmanlar
model.add(Dense(256, activation="relu"))

# iletim sönümü : 
# modelin aşırı uydurma yapmasını engeller.
# Sinir ağlarının düzleştirilmesinde kullanılır.
# verdiğimiz orana göre elemanları sıfırlar.
model.add(Dropout(0.6))

# fonksiyonu sigmoid olarak kullanarak çıkan değeri [0,1] arasına sıkıştırdık
# çünki ikili sınıflandırma var (MASK, NO MASK)
model.add(Dense(1, activation="sigmoid"))


# modelimizi görüntülemek
print(model.summary())


"""

Layer (type)                 Output Shape              Param #   
=================================================================
xception (Model)             (None, 5, 5, 2048)        20861480  
_________________________________________________________________
batch_normalization_5 (Batch (None, 5, 5, 2048)        8192      
_________________________________________________________________
flatten_1 (Flatten)          (None, 51200)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               13107456  
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 257       
=================================================================
Total params: 33,977,385
Trainable params: 17,860,609
Non-trainable params: 16,116,776

"""


# --- Modeli derleme ve eğitme --- #


# modeli derleme
model.compile(loss="binary_crossentropy", # kayıp fonksiyonu

              # eniyileme:
              # ağımızın girdisi olan veri ile oluşturduğu kaybı göz önünde
              # bulundurarak kendisini güncelleme mekanizması
			  optimizer=optimizers.RMSprop(lr=2e-5),

			  # eğitim ve test süresince takip edilecek metrikler. 
			  metrics=["acc"])


# modelin görsel halini diske kaydetme
plot_model(model, show_shapes=True, to_file="model.png")


# modeli eğitme
history = model.fit_generator( # acc, loss, val_acc, val_loss değerlerini history adlı değişkenden alacağız.

	  # eğitim verileri
      train_generator,

      # döngü bitene kadar geçeceği örnek sayısı (alınacak yığın)
      steps_per_epoch=175,

      # döngü sayısı
      epochs=100,

      # doğrulama verisi
      validation_data=validation_generator,

      # doğrulama için yığın sayısı
      validation_steps=75,

      verbose=2)

# modelimizi test için kaydettik.
model.save('mask_model.h5')



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
plt.plot(epochs, val_acc, "b", label="Doğrulama başarımı")

# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama başarımı")

plt.legend()

plt.figure()

# eğitim kaybını kendine özel çizdirdik.
plt.plot(epochs, loss, "bo", label="Eğitim kaybı")

# doğrulama kaybını kendine özel çizdirdik.
plt.plot(epochs, val_loss, "b", label="Doğrulama kaybı")


# çizdirmemizin başlığı
plt.title("Eğitim ve doğrulama kaybı")

plt.legend()

# ekrana çıkartma
plt.show()
def demir_ai_API():

  import os

  try:

    import imutils
    import cv2
    from keras.preprocessing.image import img_to_array
    from keras.models import load_model
    import wget
    import numpy

  except:

    print(" GEREKLİ MODÜLLER KURULUYOR ...")

    os.system("python3 -m pip install --upgrade pip")

    os.system("pip3 install imutils")

    #os.system("pip3 install tensorflow")

    os.system("pip3 install keras")

    os.system("pip3 install opencv-python")

    os.system("pip3 install wget")

    os.system("pip3 install numpy")

    os.system("pip3 install argparse")

    print("İŞLEM TAMAMLANDI")

  kontrol = None

  import wget

  if not os.path.isfile(os.path.join("deploy.prototxt.txt")):

    print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")

    kontrol = True 

    file_url1 = "https://www.dropbox.com/s/08hxbqqi5145v5o/deploy.prototxt.txt?dl=1"
    wget.download(file_url1)

  if not os.path.isfile(os.path.join("res10_300x300_ssd_iter_140000.caffemodel")):

    print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")

    kontrol = True

    file_url2 = "https://www.dropbox.com/s/m6v8yuymewig62n/res10_300x300_ssd_iter_140000.caffemodel?dl=1"
    wget.download(file_url2)


  if not os.path.isfile(os.path.join("resim.jpg")):

    print("\nEKSİK DOSYALAR İNDİRİLİYOR ....")

    kontrol = True

    file_url2 = "https://www.dropbox.com/s/25s94gufbnapvgy/resim.jpg?dl=1"
    wget.download(file_url2)

  if kontrol == True:

    print("\nEKSİK DOSYALAR İNDİRİLDİ")

demir_ai_API()
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import wget
import os
import matplotlib.pyplot as plt


net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel") # yüz tespiti

vs= cv2.imread("resim.jpg")
orig = vs.copy()

time.sleep(2.0)

model = load_model("mask_model.h5") # maske tespiti

frame = vs

(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
  (300, 300), (104.0, 177.0, 123.0))


net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):


  confidence = detections[0, 0, i, 2]


  if confidence < 0.5:
    continue


  box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
  (startX, startY, endX, endY) = box.astype("int")

  try:

    image = cv2.resize(frame[startY:endY, startX:endX], (150, 150)) # tespit edilen yüzü modele dahil etmek için işliyoruz
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    a = model.predict(image) # yüz modele gönderilir çıkan sonuca göre maske var yada yok

    texts = "NONE"

    if a > 0.3: # ekrana yansıtmalar
      
      text = "N O   M A S K"
      y = startY - 10 if startY - 10 > 10 else startY + 10
      cv2.rectangle(frame, (startX, startY), (endX, endY),
      (0, 0, 255), 2)
      cv2.putText(frame, text, (startX, y),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    elif a<= 0.3:

      text = "M A S K"
      y = startY - 10 if startY - 10 > 10 else startY + 10
      cv2.rectangle(frame, (startX, startY), (endX, endY),
      (0, 255, 0), 2)
      cv2.putText(frame, text, (startX, y),
      cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

  except:

    pass

plt.imshow(frame)

