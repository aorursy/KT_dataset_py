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
# libraries
#glob:class oluşturma için
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob

train_path = "../input/fruits/fruits-360/Training"
test_path = "../input/fruits/fruits-360/Test"
img = load_img(train_path + "/Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
#resmi img_to_array fonk.ile array e çevirme:
#100*100*3 lük resim:en-boy-3(rgb),renkli olduğunu belirtiyor.
x = img_to_array(img)
print(x.shape)
x
#glob fonk.ile toplam sınıfsayısını bulma.
#output shape imiz:131
#'/*' -->herhangi bir isimdeki dosyayı className e yükle
className = glob(train_path + '/*' )
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)
a=glob(train_path+"/Apple Braeburn"+"/*")
len(a)
#%% CNN Model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape = x.shape))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

batch_size = 32  #her iterasyonda 32 resmimizi train edeceğiz.
#%% Data Generation - Train - Test
train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path, 
        target_size=x.shape[:2],   #100*100(meyvelerin boyutu)
        batch_size = batch_size,   #32
        color_mode= "rgb",            
        class_mode= "categorical")   #bir den fazla classım var demek.

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")


#şimdi modeli fit etme zamanı
hist = model.fit_generator(
        generator = train_generator,  #generatorumuz,fit etmek istediğimiz datasetimiz
        steps_per_epoch = 1600 // batch_size,
        epochs=100,
        validation_data = test_generator,
        validation_steps = 800 // batch_size)

#%% model save
model.save_weights("deneme.h5")
#%% model evaluation
print(hist.history.keys()) #hist için içerinde 4 tane parametre sayısal olarak tutluyor.
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()  #labelların görünmesi için
plt.show()
plt.figure()  #resimleri ayrı ayrı göstermek için.
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()
#%% save history
import json
with open("deneme.json","w") as f:
    json.dump(hist.history, f)
#%% load history
import codecs
with codecs.open("deneme.json", "r",encoding = "utf-8") as f:
    h = json.loads(f.read())  #daha önce kaydettiğimiz historu h adlı değişkenimizie kayıt oldu
plt.plot(h["loss"], label = "Train Loss")
plt.plot(h["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(h["accuracy"], label = "Train acc")
plt.plot(h["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show() 
