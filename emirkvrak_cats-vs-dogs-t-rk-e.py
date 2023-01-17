# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-on,ly "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# çıkartma işlemi 30 sn sürebilir

import os

import zipfile



Unzipped_data = ['train_base','test_base']

[os.mkdir(path) for path in Unzipped_data]

Dataset = ['train','test']

for unzipped_data,dataset in zip(Unzipped_data,Dataset):

    path = '../input/dogs-vs-cats-redux-kernels-edition/{}.zip'.format(dataset)

    with zipfile.ZipFile(path,"r") as z:

        z.extractall(unzipped_data)
print('toplam train :' , len(os.listdir('/kaggle/working/train_base/train')))



print('Toplam test : ' , len(os.listdir('/kaggle/working/test_base/test')))

base_dir = './'

base_train_path = 'train_base/train'

base_test_path = 'test_base/test'
base_dir = './' # eğer başka bir dizinde oluşturmak istersek bunu değiştirebiliriz.

try:

    train_dir = os.path.join(base_dir,'train')

    os.mkdir(train_dir)

    

    validation_dir = os.path.join(base_dir,'validation')

    os.mkdir(validation_dir)

    

    test_dir = os.path.join(base_dir,'test')

    os.mkdir(test_dir)

    

    # üstte oluşturuğumz üç ana klasörün içine kedi ve köpek ayrı klasörler oluşturacağız

    

    train_cats_dir = os.path.join(train_dir,'cats')

    os.mkdir(train_cats_dir)

    

    train_dogs_dir = os.path.join(train_dir,'dogs')

    os.mkdir(train_dogs_dir)

    

    validation_cats_dir = os.path.join(validation_dir, 'cats')

    os.mkdir(validation_cats_dir)



    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    os.mkdir(validation_dogs_dir)



   # test_cats_dir = os.path.join(test_dir, 'cats')

   # os.mkdir(test_cats_dir)



   #test_dogs_dir = os.path.join(test_dir, 'dogs')

   #os.mkdir(test_dogs_dir)

    

except FileExistsError:

    pass
print('çalışma dizinimiz')

!ls
# kitapta söylenen veri setinin büyüklüğü çok arttığından daha fazla veri ile eğitebilirim.

# 3 bin kedi 3 bin köpek ve biner de validation alacağım.

import shutil



train_dosya_adedi = 3000

validation_dosya_adedi = 1000

test_dosya_adedi = 1000



fnames = ['cat.{}.jpg'.format(i) for i in range(train_dosya_adedi)] # ilk 3000 kedi train

for fname in fnames:

    src = os.path.join(base_train_path,fname)

    dst = os.path.join(train_cats_dir,fname)

    shutil.copyfile(src,dst)

print('kedi train dosyaları kopyalandı' ,len(os.listdir(train_cats_dir)), 'adet dosya.')



fnames = ['cat.{}.jpg'.format(i) for i in range(train_dosya_adedi,train_dosya_adedi+test_dosya_adedi)] # 3000 ile 4000 arasındakiler kedi validation

for fname in fnames:

    src = os.path.join(base_train_path,fname)

    dst = os.path.join(validation_cats_dir,fname)

    shutil.copyfile(src,dst)

print('kedi validation dosyaları kopyalandı', len(os.listdir(validation_cats_dir)), 'adet dosya.')



fnames = ['dog.{}.jpg'.format(i) for i in range(train_dosya_adedi)]

for fname in fnames:

    src = os.path.join(base_train_path,fname)

    dst = os.path.join(train_dogs_dir,fname)

    shutil.copy(src,dst)



print('köpek validation dosyaları kopyalandı', len(os.listdir(train_dogs_dir)), 'adet dosya.')



fnames = ['dog.{}.jpg'.format(i) for i in range(train_dosya_adedi,train_dosya_adedi+test_dosya_adedi)]

for fname in fnames:

    src = os.path.join(base_train_path,fname)

    dst = os.path.join(validation_dogs_dir,fname)

    shutil.copy(src,dst)

print('köpek validation dosyaları kopyalandı', len(os.listdir(validation_dogs_dir)), 'adet dosya.')



test_fnames = ['{}.jpg'. format(i) for i in range(1,test_dosya_adedi+1)]



for fname in test_fnames:

    src = os.path.join('test_base/test',fname)

    dst = os.path.join('test',fname)

    shutil.copy(src,dst)

print('test_dosyaları_kopyalandı', len(os.listdir(test_dir)))







# CNN modeli

from keras import layers

from keras import models

def build_model():

    # kanal sayısı giderek artarken 32->128 nitelik haritası boyutu ufalıyor 150x150 -> 7x7

    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (150,150,3)))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Flatten())

    

    model.add(layers.Dense(512, activation = 'relu'))

    

    model.add(layers.Dense(1, activation = 'sigmoid'))

    

    return model



my_model = build_model()
# her model oluşturduğumuzda inceletebileceğimiz bir fonksiyon yazalım 

from keras.utils.vis_utils import plot_model

%pylab inline

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from matplotlib.pyplot import figure



def examine_model(model, show_shapes = True, display_png = True):

    model.summary()

    

    print("/n")

    

    plot_model(

    model,

    to_file="model.png",

    show_shapes=show_shapes,

    show_layer_names=True,

    rankdir="TB",

    expand_nested=False,

    dpi=96,

    )

    

    if(display_png):

        figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')

        img = mpimg.imread('model.png')

        imgplot = plt.imshow(img)

        plt.show()

        

    

    

examine_model(my_model)
from keras import optimizers

# modelimizi derleyelim.

my_model.compile(loss = 'binary_crossentropy', #evet hayır seçimi olduğu için binary_crossentropy

                 optimizer = optimizers.RMSprop(), # ?

                 metrics = ['acc']

                ) 
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale = 1./255) # 1/255 ölçeklendirme

test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory('./train',

                                                   target_size =(150,150),

                                                   batch_size = 20,

                                                   class_mode = 'binary')



validation_generator = train_datagen.flow_from_directory('./validation',

                                                   target_size =(150,150),

                                                   batch_size = 20,

                                                   class_mode = 'binary')



# işlem süresinin kısalması için sağdaki üç nokta - > accelerator -> GPU

steps_per_epoch = (train_dosya_adedi * 2) / 20

validation_steps = (validation_dosya_adedi * 2) / 20

history = my_model.fit_generator(train_generator,

                             steps_per_epoch = steps_per_epoch,

                             epochs = 30,

                             validation_data = validation_generator,

                             validation_steps = validation_steps )


acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='egitim basarimi')

plt.plot(epochs, val_acc, 'b', label='dogrulama basarimi')

plt.title('egitim ve dogrulama basarimi')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='egitim kaybı')

plt.plot(epochs, val_loss, 'b', label='dogrulama kaybı')

plt.title('Egitim ve dogrulama kaybı')

plt.legend()



plt.show()
datagen = ImageDataGenerator(

      rotation_range=40, # derece cinsinden rastgele döndürme açısı

      width_shift_range=0.2, # yatay ve dikey kaydırma oranları

      height_shift_range=0.2,

      shear_range=0.2, # burkma 

      zoom_range=0.2, # yakınlaştırma

      horizontal_flip=True, # dikeyde resmi döndürme

      fill_mode='nearest') # ortaya çıkan fazla görüntü noktalarını doldurma
# ön işleme yapacağımız modülü import edelim

from keras.preprocessing import image



fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]





img_path = fnames[11] #rasgele bir dosya 





img = image.load_img(img_path, target_size=(150, 150))





x = image.img_to_array(img)





x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0: # 4 adet üretiyoruz.

        break



plt.show()
def build_model():

    # dense katmanlarından hemen önce bir dropout ekleyelim

    model = models.Sequential()

    model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (150,150,3)))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

    model.add(layers.MaxPooling2D((2,2)))

    

    model.add(layers.Flatten())

    

    model.add(layers.Dropout(0.5)) # !!

    

    model.add(layers.Dense(512, activation = 'relu'))

    

    model.add(layers.Dense(1, activation = 'sigmoid'))

    

    return model



my_model = build_model()



my_model.compile(loss = 'binary_crossentropy' , optimizer = optimizers.RMSprop(), metrics = ['acc'])

# train ve validation generator olusturalım. 



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        './train',

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=32,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        './validation',

        target_size=(150, 150),

        batch_size=32,

        class_mode='binary')



history = my_model.fit_generator(

      train_generator,

      steps_per_epoch=100,

      epochs=100,

      validation_data=validation_generator,

      validation_steps=50)

 #tekrar çizdirelim

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='egitim basarimi')

plt.plot(epochs, val_acc, 'b', label='dogrulama basarimi')

plt.title('egitim ve dogrulama basarimi')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='egitim kaybı')

plt.plot(epochs, val_loss, 'b', label='dogrulama kaybı')

plt.title('Egitim ve dogrulama kaybı')

plt.legend()

# modeli kaydedelim

model.save('cats_and_dogs_v1.h5')