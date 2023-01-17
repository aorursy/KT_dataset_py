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
#%% Libraries import

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img #preprocessing for image

from keras.models import Sequential #Model oluşturmak

from keras.layers import Dense #Model için

from keras.applications.vgg16 import VGG16 #for transfer learning

import matplotlib.pyplot as plt #görsellşetirme

from glob import glob #f o r image import
train_path = "../input/fruits/fruits-360/Training"

test_path = "/kaggle/input/fruits/fruits-360/Test"



img = load_img('../input/fruits/fruits-360/Training/Avocado/0_100.jpg')

plt.imshow(img)

plt.axis("off")

plt.show()



x = img_to_array(img)

print(x.shape)





numberOfClass = len(glob('../input/fruits/fruits-360/Training'+'/*')) #CLass sayısını klasör içine gidip okuyoruz. * koyarak hepsini okumasını sağlıyoruz. Böylece class sayısı otomatik olarak belirleniyor.
vgg = VGG16()
print(vgg.summary())

print(type(vgg))



vgg_layer_list = vgg.layers
model = Sequential() 

for i in range(len(vgg_layer_list)-1): 

    model.add(vgg_layer_list[i]) 

print(model.summary())
for layers in model.layers: #Burada eklenilen layerlarını train edilebilme özelliklerini kapatıyoruz. Çünkü zaten bunlar ileri derecede eğitimli. (Transfer Learning is here!)

    layers.trainable = False



model.add(Dense(numberOfClass, activation="softmax")) #Vgg16'daan çıkardığımız dense layerı kendi datamıza uygun olarak ekliyoruz.



print(model.summary())



model.compile(loss = "categorical_crossentropy",

              optimizer = "rmsprop",

              metrics = ["accuracy"]) #compliting 



train_data = ImageDataGenerator().flow_from_directory(train_path,target_size = (224,224)) 

test_data = ImageDataGenerator().flow_from_directory(test_path,target_size = (224,224)) 



batch_size = 32



hist = model.fit_generator(train_data,

                           steps_per_epoch=1600//batch_size,

                           epochs= 25,

                           validation_data=test_data,

                           validation_steps= 800//batch_size,)
model.save_weights("./weights.h5")
plt.title('Loss Scores')

print(hist.history.keys())

plt.plot(hist.history["loss"],label = "training loss")

plt.plot(hist.history["val_loss"],label = "validation loss")

plt.legend()

plt.show()

plt.figure()

plt.title('Accuracy Scores')

plt.plot(hist.history["accuracy"],label = "training acc")

plt.plot(hist.history["val_accuracy"],label = "validation acc")

plt.legend()

plt.show()
