# first thing is extracting the files

import os, shutil, zipfile



data = ['train', 'test1']



for el in data:

    with zipfile.ZipFile('../input/dogs-vs-cats/' + el + ".zip", "r") as z:

        z.extractall(".")
#shutil.rmtree('/kaggle/working/small') # deletes the 'small' directory, for debugging purposes
path = os.getcwd()

print ("The current working directory is %s" % path)



original_dataset_dir = '/kaggle/working' # where the data was extracted

base_dir = '/kaggle/working/small' # directory for the smaller dataset we'll be using (just for less runtime)

os.mkdir(base_dir)



train_dir = os.path.join(base_dir, 'train')

os.mkdir(train_dir)

validation_dir = os.path.join(base_dir, 'validation')

os.mkdir(validation_dir)

test_dir = os.path.join(base_dir, 'test')

os.mkdir(test_dir)



train_cats_dir = os.path.join(train_dir, 'cats')

os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')

os.mkdir(train_dogs_dir)



validation_cats_dir = os.path.join(validation_dir, 'cats')

os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')

os.mkdir(validation_dogs_dir)



test_cats_dir = os.path.join(test_dir, 'cats')

os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')

os.mkdir(test_dogs_dir)
#print(os.listdir("/kaggle/working/train/"))



original_dataset_dir = '/kaggle/working/train' # for the smaller sample size we use only part of the training set



# out of the total 25'000 images of cats n dogs (12'500 for each class), we'll be using, for each class:

# just 1'000 images for training, 500 for validation and 500 for testing

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_cats_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_cats_dir, fname)

    shutil.copyfile(src, dst)



fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_cats_dir, fname)

    shutil.copyfile(src, dst)

    

# dogs

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(train_dogs_dir, fname)

    shutil.copyfile(src, dst)

    

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(validation_dogs_dir, fname)

    shutil.copyfile(src, dst)



fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]

for fname in fnames:

    src = os.path.join(original_dataset_dir, fname)

    dst = os.path.join(test_dogs_dir, fname)

    shutil.copyfile(src, dst)
# since we're having a binary classification problem, the network will end with a single unit (Dense layer size 1) with a sigmoid function

from keras import models

from keras import layers

from keras import optimizers



model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()



# since we ended the network with a single layer, the loss will be calculted using binary crossentropy

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator

# contains generators that can be used to automatically convert images into tensors

# the generator object acts as an iterator and works with the yield operator:

def generator():

    i = 0

    while True:

        i += 1

        yield i

        

# creating the generators

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(train_dir, # img

                                                    target_size=(150,150), # resize images to 150x150

                                                    batch_size=20,

                                                    class_mode='binary') # using binary crossentropy calls for binary labels



validation_generator = test_datagen.flow_from_directory(validation_dir,

                                                       target_size=(150,150),

                                                       batch_size=20,

                                                       class_mode='binary')

                                                    

for data_batch, labels_batch in train_generator:

    print('data batch shape: ', data_batch.shape) # (20, 150, 150, 3)

    print('label batch shape: ', labels_batch.shape) # (20,)

    break # because it generates the data indefinetely

    



# we'll fit the model using the data generator through the fit_generator method. it expects as its first input a generator that yields batches of inputs and targets

# steps_per_epoch is an argument that determines how many batches are required for an epoch to pass

# after steps_per_epoch gradient descent steps we go to the next epoch (each batch is a gradient descent step). since each batch is 20 samples, after 100 we'll have the data

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,

                             validation_data=validation_generator, validation_steps=50)



model.save('cats_vs_dogs.h5')
import matplotlib.pyplot as plt



plt.style.use('ggplot')



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'b', label='training acc')

plt.plot(epochs, val_acc, 'r', label='validation acc')

plt.title('accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'b', label='training loss')

plt.plot(epochs, val_loss, 'r', label='validation loss')

plt.title('loss')

plt.legend()

plt.show()
datagen = ImageDataGenerator(rotation_range=40, # range 0-180 within which to rotate images

                            width_shift_range=0.2,

                            height_shift_range=0.2, # randomly translate pictures vertically and horizontally

                            shear_range=0.2, # shearing transformations (kinda like, making the picture at an angle)

                            zoom_range=0.2,

                            horizontal_flip=True,

                            fill_mode='nearest') # how to fill new pixels



# just to be able to look at the images with the transformations:

from keras.preprocessing import image # module for image processing



fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[420] # the example image chosen for augmentation

img = image.load_img(img_path, target_size=(150,150)) # resizing just like before



x = image.img_to_array(img) # to shape (150,150,3)

x = x.reshape((1,) + x.shape)



i = 0

for batch in datagen.flow(x, batch_size=1): # cuz we just wanna demo the transformations of 1 pic

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 3 == 0:

        break

        

plt.show()



# while this helps, the inputs will be heavily correlated, to ensure that we get a smaller overfit we'll add a dropout layer right before the Dense layer:

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# finally, we'll train the network using the data augmentation and dropout:

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255) # the test/validation data shall not be augmented



train_generator = train_datagen.flow_from_directory(train_dir,

                                                   target_size=(150,150),

                                                   batch_size=32,

                                                   class_mode='binary')



validation_generator = test_datagen.flow_from_directory(validation_dir,

                                                       target_size=(150,150),

                                                       batch_size=32,

                                                       class_mode='binary')



history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100,

                             validation_data=validation_generator, validation_steps=50)



model.save('cats_vs_dogs_aug.h5')



# we get way better results, but an even better way to improve the accuracy is by using a pretrained model :)