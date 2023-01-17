import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Input
from keras.regularizers import l2
from keras.optimizers import rmsprop
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import gc
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def fn(file_path):       # 1.Get file names from directory
    file_list=os.listdir(file_path)
    file_list=np.array(file_list)
    return file_list

root_train_cat="/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/train/cat/"
root_train_dog="/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/train/dog/"
root_validation_cat="/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/validation/cat/"
root_validation_dog="/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/validation/dog/"
def read_image_to_array(root,file):
  train_image=image.load_img(root+file,grayscale=True,color_mode = "grayscale",target_size=(112,112))
  image_pixels= image.img_to_array(train_image)
  image_pixels = np.array(image_pixels)
  return image_pixels
x_train_cat=[]
y_train_cat=[]
file_list=fn("/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/train/cat/")
for ind in file_list:
    file_name = ind
    pixels=read_image_to_array(root_train_cat,file_name)
    x_train_cat.append(pixels)
    img_type=[ind][0].split(".")
    if img_type[0]=="cat":
        img_type[0]="0"
    ctt=tf.keras.utils.to_categorical(img_type[0], num_classes=2)
    y_train_cat.append(ctt)
len(x_train_cat)
x_train_dog=[]
y_train_dog=[]
file_list=fn("/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/train/dog/")
for ind in file_list:
    file_name = ind
    pixels=read_image_to_array(root_train_dog,file_name)
    x_train_dog.append(pixels)
    img_type=[ind][0].split(".")
    if img_type[0]=="dog":
        img_type[0]="1"
    ctt=tf.keras.utils.to_categorical(img_type[0], num_classes=2)
    y_train_dog.append(ctt)
len(x_train_dog)
x_train=x_train_cat + x_train_dog
y_train=y_train_cat + y_train_dog
print(len(x_train))
print(len(y_train))
x_validation_cat=[]
y_validation_cat=[]
file_list=fn("/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/validation/cat/")
for ind in file_list:
    file_name = ind
    pixels=read_image_to_array(root_validation_cat,file_name)
    x_validation_cat.append(pixels)
    img_type=[ind][0].split(".")
    if img_type[0]=="cat":
        img_type[0]="0"
    ctt=tf.keras.utils.to_categorical(img_type[0], num_classes=2)
    y_validation_cat.append(ctt)
len(x_validation_cat)
x_validation_dog=[]
y_validation_dog=[]
file_list=fn("/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/validation/dog/")
for ind in file_list:
    file_name = ind
    pixels=read_image_to_array(root_validation_dog,file_name)
    x_validation_dog.append(pixels)
    img_type=[ind][0].split(".")
    if img_type[0]=="dog":
        img_type[0]="1"
    ctt=tf.keras.utils.to_categorical(img_type[0], num_classes=2)
    y_validation_dog.append(ctt)
len(x_validation_dog)
x_validation=x_validation_cat + x_validation_dog
y_validation=y_validation_cat + y_validation_dog
len(x_validation)
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_validation = np.array(x_validation, 'float32')
y_validation = np.array(y_validation, 'float32')

x_validation /= 255
x_train /= 255



x_train = x_train.astype('float32')
x_validation = x_validation.astype('float32')
import gc
from keras.callbacks import CSVLogger
batch_size = 40
epochs = 65
gc.collect()
alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=(112, 112, 1), padding='same', kernel_regularizer=l2(0.002)))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(256, (5, 5), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(512, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))

# Layer 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
alexnet.add(Dense(3072))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.7))

# Layer 7
alexnet.add(Dense(4096))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.7))

# Layer 8
alexnet.add(Dense(2))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

#Batch (Küme) işlemleri
gen = ImageDataGenerator()
gen.fit(x_train)
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)

alexnet.compile(loss='categorical_crossentropy'
    , optimizer=rmsprop(lr=0.0001, decay=1e-6)
    , metrics=['accuracy']
)

fit = True #Train etmek için fit=True olarak değiştiriniz.

if fit == True:
    #alexnet.fit_generator(x_train, y_train, epochs=epochs, callbacks=callbacks, verbose=1) #Tüm veri kümesi için eğit
    alexnet.fit_generator(train_generator, validation_data=(x_validation, y_validation), steps_per_epoch=batch_size, epochs=epochs, verbose=1,shuffle=True, callbacks=[CSVLogger('training_animals.log', separator=',', append=False)])
    alexnet.save('animals.h5')

else:
    alexnet.load_weights('/kaggle/working/trained_h5/animals.h5') #Öğrenilmiş ağırlıkları yükle
    alexnet.history = pd.read_csv('training_animals.log', sep=',', engine='python')
def loss_accuracy_graph(history):
    loss_history = history.history['loss']
    acc_history = history.history['accuracy']
    val_loss_history = history.history['val_loss']
    val_acc_history = history.history['val_accuracy']
    epoch = [(i + 1) for i in range(epochs)]

    plt.rcParams.update({'font.size': 12})
    plt.plot(epoch, loss_history, color='red', label='Eğitim Hatası')
    plt.plot(epoch, val_loss_history, color='orange', label='Test Hatası')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.title('Hata Grafiği\n')
    plt.legend()
    plt.show()
    plt.close()
    
    plt.plot(epoch, acc_history, color='green', label='Eğitim Başarısı')
    plt.plot(epoch, val_acc_history, color='blue', label='Test Başarısı')
    plt.xlabel('Epoch')
    plt.ylabel('Kazanç')
    plt.title('Kazanç Grafiği\n')
    plt.legend()
    plt.show()
    plt.close()
if fit == True:
    loss_accuracy_graph(alexnet.history)
else:
    loss_accuracy_graph(alexnet)
image_t= image.load_img("/kaggle/input/dogs-vs-cats-train-validadion-and-evaluation/evaluation/dog/12.jpg",grayscale=True,target_size=(112,112))

pixels= image.img_to_array(image_t)
pixels = np.expand_dims(pixels, axis = 0)

pixels /= 255

custom = alexnet.predict(pixels)

y_pos = np.arange(2)

plt.rcParams.update({'font.size': 12})
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='g')
plt.xticks(y_pos, ["cat","dog"], rotation='vertical')
plt.ylabel('Benzerlik (%)')
plt.title('Hayvan')
#plt.savefig('books_read.png', dpi= 2400)
plt.show()