!ls ../input/dog-breed-identification
import numpy as np



original_train_dir = '../input/dog-breed-identification/train'

original_test_dir = '../input/dog-breed-identification/test'

train_labels = np.loadtxt('../input/dog-breed-identification/labels.csv', delimiter=',', dtype=str, skiprows=1)

# Remove missing data, this image was missing on my dataset?

# train_labels = train_labels[train_labels[:, 0] != '000bec180eb18c7604dcecc8fe0dba07']

clazzes, counts = np.unique(train_labels[:, 1], return_counts=True)

print("Some classes with count:")

print(np.asarray((clazzes, counts)).T[0:10])

print("Number of class: %d" % clazzes.size)
import os, shutil



def mkdirIfNotExist(directory):

    if not os.path.exists(directory):

        os.mkdir(directory)

    return directory



base_dir = mkdirIfNotExist('./data_gen')

train_dir = mkdirIfNotExist(os.path.join(base_dir, 'train'))

validation_dir = mkdirIfNotExist(os.path.join(base_dir, 'validation'))

test_dir = mkdirIfNotExist(os.path.join(base_dir, 'test'))

for clazz in clazzes[:]:

    mkdirIfNotExist(os.path.join(train_dir, clazz))

    mkdirIfNotExist(os.path.join(validation_dir, clazz))
def copyIfNotExist(fnames, src_dir, dst_dir):

    nCopied = 0

    for fname in fnames:

        src = os.path.join(src_dir, fname)

        dst = os.path.join(dst_dir, fname)

        if not os.path.exists(dst):

            shutil.copyfile(src, dst)

            nCopied += 1

    if nCopied > 0:

        print("Copied %d to %s" % (nCopied, dst_dir))



# This will split available labeled data to train-validation sets

train_ratio = 0.7

for clazz in clazzes[:]:

    fnames = train_labels[train_labels[:, 1] == clazz][:,0]

    fnames = ['{}.jpg'.format(name) for name in fnames]

    idx = int(len(fnames)*(1-train_ratio))

    val_fnames = fnames[:idx]

    train_fnames = fnames[idx:]

    train_class_dir = os.path.join(train_dir, clazz)

    validation_class_dir = os.path.join(validation_dir, clazz)

    copyIfNotExist(train_fnames, original_train_dir, train_class_dir)

    copyIfNotExist(val_fnames, original_train_dir, validation_class_dir)
from keras.preprocessing.image import ImageDataGenerator

img_width ,img_height = 299, 299

batch_size = 16



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    shear_range=0.1,

    zoom_range=0.1

)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical')

total_train_image_count = train_generator.samples

class_count = train_generator.num_class



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(img_width, img_height),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=False)

total_val_image_count = train_generator.samples
from keras.preprocessing import image

import matplotlib.pyplot as plt



train_first_dir = os.path.join(train_dir, clazzes[0])

fnames = [os.path.join(train_first_dir, fname) for fname in os.listdir(train_first_dir)]



img_path = fnames[3]

img = image.load_img(img_path, target_size=(img_width, img_height))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)



i = 0

for batch in train_datagen.flow(x, batch_size=1):

    plt.figure(i)

    imgplot = plt.imshow(image.array_to_img(batch[0]))

    i += 1

    if i % 4 == 0:

        break



plt.show()
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.mkdir(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.mkdir(models_dir)
!cp ../input/keras-pretrained-models/* ~/.keras/models/
!ls ~/.keras/models
from keras.applications.xception import Xception



conv_base = Xception(weights='imagenet',

                     include_top=False,

                     input_shape=(img_width, img_height, 3))

conv_base.trainable = False
from keras import layers, models, regularizers, optimizers

from keras.models import Sequential,  Model

from keras.layers import Flatten, Dense, Dropout



model = models.Sequential()

model.add(conv_base)

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(class_count, activation='sigmoid'))



model.compile(loss='categorical_crossentropy',

              optimizer=optimizers.SGD(lr=1e-4, momentum=0.90),

              metrics=['acc'])

model.summary()
from time import strftime



history = model.fit_generator(

      train_generator,

##       steps_per_epoch=int(total_train_image_count / batch_size),

      steps_per_epoch=1,

      epochs=1,

      validation_data=validation_generator,

##      validation_steps=int(total_val_image_count / batch_size)

      validation_steps=1

)



# time_str = strftime("%Y%m%d_%H%M%S")

# model.save('dog_breed_pretrain_xception_299_{}.h5py'.format(time_str))
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo')

plt.plot(epochs, val_acc, 'b')

plt.title('Training and validation accuracy')



plt.figure()



plt.plot(epochs, loss, 'bo')

plt.plot(epochs, val_loss, 'b')

plt.title('Training and validation loss')



plt.show()
from keras.preprocessing import image

import numpy as np



def load_test_image(fpath):

    img = image.load_img(fpath, target_size=(img_width, img_height))

    x = image.img_to_array(img)

    return x



test_labels = np.loadtxt('../input/dog-breed-identification/sample_submission.csv', delimiter=',', dtype=str, skiprows=1)

test_images = []

test_names = test_labels[:,0]

# Slow on Kaggle server

#for test_name in test_names:

#    fname = '{}.jpg'.format(test_name)

#    data = load_test_image(os.path.join(original_test_dir, fname))

#    test_images.append(data)



test_images = np.asarray(test_images)

test_images = test_images.astype('float32')

test_images /= 255

print(test_images.shape)
# Slow on Kaggle server

# predictions = model.predict(test_images, verbose=1)
import pandas as pd

class_indices = sorted([ [k,v] for k, v in train_generator.class_indices.items() ], key=lambda c : c[1])

columns = [b[0] for b in class_indices]

# No prediction, no

# df = pd.DataFrame(predictions,columns=columns)

# df = df.assign(id = test_names)

# print(df.head())



# df.to_csv("submit.csv", index=False)