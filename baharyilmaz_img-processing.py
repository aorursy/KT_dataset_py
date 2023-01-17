import keras
keras.__version__
# Veriseti 
import os
dataset=os.listdir('/kaggle/input/')
print(dataset)
print('Train için bisiklet verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Train/bike')))
print('Validation için bisiklet verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Validation/bike')))
print('Test için bisiklet verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Test/bike')))
print('\n')
print('Train için güneş gözlüğü verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Train/sunglasses')))
print('Validation için güneş gözlüğü verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Validation/sunglasses')))
print('Test için güneş gözlüğü verisi:',len(os.listdir('/kaggle/input/bikesunglasses/Test/sunglasses')))

# Kütüphanelerin çağırılması

from keras import layers
from keras import models
# derin öğrenme pipeline'i, sıralı katmanlardan oluşan derin öğrenme modeli

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(160, 92, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator


data_train='/kaggle/input/bikesunglasses/Train'
data_validation='/kaggle/input/bikesunglasses/Validation'


# resim verisini gray-scale olarak ayarlama
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        data_train,
        target_size=(160,92),# resim verisini yeniden boyutlandırma
        batch_size=10,# veriseti 20 parçaya bölme, her seferde verisetinin bir parçası ile eğitim yapılıcak
        class_mode="binary") # ikili sınıflandırma için

validation_generator = validation_datagen.flow_from_directory(
        data_validation,
        target_size=(160,92),
        batch_size=10,
        class_mode="binary")
history = model.fit_generator(
      train_generator, # eğitim verisi
      steps_per_epoch=20,
      epochs=40,            
      validation_data=validation_generator,
      validation_steps=5)
history_dict=history.history
history_dict.keys()
model.save('model.h5')
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# Modeli daha önce görmediği verilerle test etme

from keras.preprocessing.image import ImageDataGenerator

data_test='/kaggle/input/bikesunglasses/Test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_test,
        target_size=(160, 92),
        batch_size=5,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=5)
print('test acc:', test_acc)
print('test loss:', test_loss)
# yeniden ölçeklendirme özellikleri
datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')
import matplotlib.pyplot as plt
from keras.preprocessing import image


# bir resim üzerinde data augmentation sonuçlarını görme

train_bike='/kaggle/input/bikesunglasses/Train/bike'

fnames = [os.path.join(train_bike, fname) for fname in os.listdir(train_bike)]

img_path = fnames[4]

img = image.load_img(img_path, target_size = (160, 92))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1):
    plt.figure()
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

# validation için aynı işlemi uygulamıyoruz
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (160, 92, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

data_train='/kaggle/input/bikesunglasses/Train'
data_validation='/kaggle/input/bikesunglasses/Validation'

# resim verisini yeniden boyutlandırma

train_generator = train_datagen.flow_from_directory(
                                data_train,
                                target_size = (160, 92),
                                batch_size = 10,
                                class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                                data_validation,
                                target_size = (160, 92),
                                batch_size = 5,
                                class_mode = 'binary')
# modelin eğitilmesi

history = model.fit_generator(
                            train_generator,
                            steps_per_epoch = 20,
                            epochs = 40,
                            validation_data = validation_generator,
                            validation_steps = 5)
model.save("model2.h5")
# Eğitim ve validasyon sonuçlarının görselleştirilmesi

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# Modelimizi daha önce görmediği verilerle test ediyoruz

from keras.preprocessing.image import ImageDataGenerator

data_test='/kaggle/input/bikesunglasses/Test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_test,
        target_size=(160, 92),
        batch_size=5,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=5)
print('test acc:', test_acc)
print('test loss:', test_loss)
# Modelimizi tekrar oluşturuyoruz

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (160, 92, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

model.summary()
# data augmentation işlemi
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


data_train='/kaggle/input/bikesunglasses/Train'
data_validation='/kaggle/input/bikesunglasses/Validation'

# resimleri yeniden boyutlandırma
train_generator = train_datagen.flow_from_directory(
                                data_train,
                                target_size = (160, 92),
                                batch_size = 10,
                                class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                                data_validation,
                                target_size = (160, 92),
                                batch_size = 5,
                                class_mode = 'binary')
# modeli eğitme
history = model.fit_generator(
                            train_generator,
                            steps_per_epoch = 20,
                            epochs = 40,
                            validation_data = validation_generator,
                            validation_steps = 5)
model.save("model3.h5")
# Eğitim ve validasyon sonuçlarının görselleştirilmesi

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
# Modeli test etme

from keras.preprocessing.image import ImageDataGenerator

data_test='/kaggle/input/bikesunglasses/Test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_test,
        target_size=(160, 92),
        batch_size=5,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=5)
print('test acc:', test_acc)
print('test loss:', test_loss)
# Modelimizi tekrar oluşturuyoruz

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (160, 92, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Drop out
model.add(layers.Dropout(0.5))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])

# Data augmentation işlemi
train_datagen = ImageDataGenerator(
                rescale = 1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


data_train='/kaggle/input/bikesunglasses/Train'
data_validation='/kaggle/input/bikesunglasses/Validation'

# resimleri yeniden boyutlandırma
train_generator = train_datagen.flow_from_directory(
                                data_train,
                                target_size = (160, 92),
                                batch_size = 10,
                                class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
                                data_validation,
                                target_size = (160, 92),
                                batch_size = 5,
                                class_mode = 'binary')

# Daha düşük bir epoch sayısı ile modelimizi eğitiyoruz
history = model.fit_generator(
                            train_generator,
                            steps_per_epoch = 20,
                            epochs = 9,
                            validation_data = validation_generator,
                            validation_steps = 5)
model.save("model4.h5")
# Eğitim ve validasyon sonuçlarını görselleştiriyoruz 

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.show()
# Modelimizi daha önce görmediği verilerle test ediyoruz

from keras.preprocessing.image import ImageDataGenerator

data_test='/kaggle/input/bikesunglasses/Test'

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        data_test,
        target_size=(160, 92),
        batch_size=5,
        class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=5)
print('test acc:', test_acc)
print('test loss:', test_loss)