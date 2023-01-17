import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import tensorflow



from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.random import set_random_seed

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Flatten, Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam

from tensorflow.keras.regularizers import l1, l2, L1L2

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



set_random_seed(0)

np.random.seed(0)
path = "../input/split-garbage-dataset/split-garbage-dataset"
train_datagen = ImageDataGenerator(

        rescale = 1./255,

        rotation_range = 20,

        width_shift_range = 0.2,

        height_shift_range = 0.2,

        horizontal_flip = True,

        vertical_flip = True,

        fill_mode='nearest'

)

validation_datagen = ImageDataGenerator(

        rescale = 1./255

)

test_datagen = ImageDataGenerator(

        rescale = 1./255

)
img_shape = (224, 224, 3) # default values



train_batch_size = 256

val_batch_size = 32



train_generator = train_datagen.flow_from_directory(

            path + '/train',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = train_batch_size,

            class_mode = 'categorical',)



validation_generator = validation_datagen.flow_from_directory(

            path + '/valid',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            shuffle=False)



test_generator = test_datagen.flow_from_directory(

            path + '/test',

            target_size = (img_shape[0], img_shape[1]),

            batch_size = val_batch_size,

            class_mode = 'categorical',

            shuffle=False,)
vgg = VGG16(weights = 'imagenet',

              include_top = False,

              input_shape = img_shape)
# Freeze the layers except the last 3 layers

for layer in vgg.layers[:-3]:

    layer.trainable = False
# Create the model

model = Sequential()

 

# Add the vgg convolutional base model

model.add(vgg)

 

# Add new layers

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))
model.summary()
# Compile the model

model.compile(loss='categorical_crossentropy',

              optimizer=Nadam(lr=1e-4),

              metrics=['acc'])
# Train the model

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

mc = ModelCheckpoint('VGG16 Garbage Classifier.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)



history = model.fit_generator(

    train_generator,

    steps_per_epoch=train_generator.samples/train_generator.batch_size ,

    epochs=30,

    validation_data=validation_generator,

    validation_steps=validation_generator.samples/validation_generator.batch_size,

    verbose=0,

    callbacks = [es, mc],)
train_acc = history.history['acc']

val_acc = history.history['val_acc']

train_loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(train_acc) + 1)



plt.plot(epochs, train_acc, 'b*-', label = 'Training accuracy')

plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, train_loss, 'b*-', label = 'Training loss')

plt.plot(epochs, val_loss, 'r', label = 'Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
data = np.load('../input/test-data/test_data.npz')

x_test, y_test = data['x_test'], data['y_test']

y_pred = model.predict(x_test)
acc = np.count_nonzero(np.equal(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)))/x_test.shape[0]

print(acc)
confusion_matrix = np.zeros((6,6), dtype=np.uint8)

per_class_acc = np.zeros(6)

for i in range(y_test.shape[1]):

    idxs = np.argmax(y_test, axis=1)==i

    this_label = y_test[idxs]

    num_samples_per_class = np.count_nonzero(idxs)

    one_hot = tensorflow.one_hot(np.argmax(model.predict(x_test[idxs]), axis=1), depth=6).eval(session=tensorflow.Session())

    confusion_matrix[i] = np.sum(one_hot, axis=0)

    per_class_acc[i] = confusion_matrix[i,i]/num_samples_per_class
LABELS=['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

plt.figure(figsize=(10,8))

sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt = 'd')

plt.title('Confusion Matrix')

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
dict(zip(LABELS, per_class_acc))