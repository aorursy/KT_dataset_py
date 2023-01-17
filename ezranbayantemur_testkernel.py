import os, shutil

import keras

keras.__version__
original_dataset_dir = '../input/photos/photos'

os.listdir(original_dataset_dir)
base_dir = '../workspace'

os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



# Train klasörü içine kalemler klasörü açılıyor

train_kalem_dir = os.path.join(train_dir, 'kalem')

os.mkdir(train_kalem_dir)



# Train klasörü içine telefonlar klasörü açılıyor

train_bardak_dir = os.path.join(train_dir, 'bardak')

os.mkdir(train_bardak_dir)



# Validation klasörü içine kalemler klasörü açılıyor

validation_kalem_dir = os.path.join(validation_dir, 'kalem')

os.mkdir(validation_kalem_dir)



# Validation klasörü içine telefonlar klasörü açılıyor

validation_bardak_dir = os.path.join(validation_dir, 'bardak')

os.mkdir(validation_bardak_dir)



# Test klasörü içine kalemler klasörü açılıyor

test_kalem_dir = os.path.join(test_dir, 'kalem')

os.mkdir(test_kalem_dir)



# Test klasörü içine telefonlar klasörü açılıyor

test_bardak_dir = os.path.join(test_dir, 'bardak')

os.mkdir(test_bardak_dir)
#Kalem fotoğraflarının kopyalanması

fnames = ['kalem ({}).JPG'.format(i) for i in range(60)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname) #Kaynaktan dosyayı seç

    dst = os.path.join(train_kalem_dir, fname) #Hedef konum seç

    shutil.copyfile(src, dst) #Kopyala



fnames = ['kalem ({}).JPG'.format(i) for i in range(60, 80)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_kalem_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['kalem ({}).JPG'.format(i) for i in range(80, 100)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_kalem_dir, fname)

    shutil.copyfile(src, dst)

    

#Bardak fotoğraflarının kopyalanması   

fnames = ['bardak ({}).JPG'.format(i) for i in range(60)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname) #Kaynaktan dosyayı seç

    dst = os.path.join(train_bardak_dir, fname) #Hedef konum seç

    shutil.copyfile(src, dst) #Kopyala



fnames = ['bardak ({}).JPG'.format(i) for i in range(60, 80)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_bardak_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['bardak ({}).JPG'.format(i) for i in range(80, 100)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_bardak_dir, fname)

    shutil.copyfile(src, dst)
print('Train: kalem fotoğrafı:', len(os.listdir(train_kalem_dir)))

print('Train: bardak fotoğrafı:', len(os.listdir(train_bardak_dir)))

print('Validation: kalem fotoğrafı:', len(os.listdir(validation_kalem_dir)))

print('Validation: bardak fotoğrafı:', len(os.listdir(validation_bardak_dir)))

print('Test: kalem fotoğrafı:', len(os.listdir(test_kalem_dir)))

print('Test: bardak fotoğrafı:', len(os.listdir(test_bardak_dir)))
from keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dir,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=20,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary'

)



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary'

)
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(8, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
history = model.fit_generator(

      train_generator,

      train_generator.n/train_generator.batch_size,

      epochs=30,

      validation_data=validation_generator,

      validation_steps=50)
model.save('kalem-bardak_1.h5')
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
#DATA AUGMENTATION

datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
# This is module with image preprocessing utilities

from keras.preprocessing import image

import matplotlib.pyplot as plt



fnames = [os.path.join(train_bardak_dir, fname) for fname in os.listdir(train_bardak_dir)]



# We pick one image to "augment"

img_path = fnames[3]



# Read the image and resize it

img = image.load_img(img_path, target_size=(150, 150))



# Convert it to a Numpy array with shape (150, 150, 3)

x = image.img_to_array(img)



# Reshape it to (1, 150, 150, 3)

x = x.reshape((1,) + x.shape)



# The .flow() command below generates batches of randomly transformed images.

# It will loop indefinitely, so we need to `break` the loop at some point!

i = 0

for batch in datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(16, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(8, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)



# Note that the validation data should not be augmented!

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_dir,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=32,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(150, 150),

        batch_size=32,

        class_mode='binary')
history = model.fit_generator(

      train_generator,

      train_generator.n/train_generator.batch_size,

      epochs=30,

      validation_data=validation_generator,

      validation_steps=50)
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