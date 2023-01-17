import numpy as np

import matplotlib.pyplot as plt

from os import listdir



from keras import Model, backend as K

from keras.layers import *

from keras.optimizers import SGD

from keras.initializers import Constant

from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
train_dir = '../input/dataset1/dataset1/train/'

test_dir = '../input/dataset1/dataset1/test/'

valid_dir = '../input/dataset1/dataset1/valid/'

alpha = 1 # Variable used to decay dropout and data augmentation.

epochs = 100

batch_size = 128

size = 100

seed = 1234
def model():

    x = Input((size, size, 3))

    

    y = Conv2D(32, 3, activation='relu')(x)

    y = BatchNormalization()(y)

    y = Conv2D(32, 3, activation='relu')(y)

    y = MaxPool2D()(y)

    

    y = BatchNormalization()(y)

    y = Conv2D(64, 3, activation='relu')(y)

    y = BatchNormalization()(y)

    y = Conv2D(64, 3, activation='relu')(y)

    y = MaxPool2D()(y)

    

    y = BatchNormalization()(y)

    y = Conv2D(128, 3, activation='relu')(y)

    y = BatchNormalization()(y)

    y = Conv2D(128, 3, activation='relu')(y)

    y = BatchNormalization()(y)

    y = Conv2D(128, 3, activation='relu')(y)

    

    y = GlobalAvgPool2D()(y)

    y = Dense(256, activation='relu')(y)

    y = Dense(1, activation='sigmoid')(y)

    return Model(x, y)



model = model()

model.summary()

model.compile(SGD(momentum=.9, nesterov=True), 'binary_crossentropy', metrics=['accuracy'])
train_loss = []

test_loss = []

train_acc = []

test_acc = []

best_acc = 0



for e in range(epochs):

    train_idg = ImageDataGenerator(rotation_range = .25*alpha,

                                   width_shift_range = .2*alpha,

                                   height_shift_range = .2*alpha,

                                   zoom_range = .2*alpha,

                                   horizontal_flip = True,

                                   preprocessing_function = lambda x: x/127.5-1)

    train_idg = train_idg.flow_from_directory(train_dir,

                                              (size, size),

                                              class_mode = 'binary',

                                              batch_size = batch_size,

                                              seed = seed)

    

    test_idg = ImageDataGenerator(preprocessing_function = lambda x: x/127.5-1)

    test_idg = test_idg.flow_from_directory(test_dir,

                                            (size, size),

                                            class_mode = 'binary',

                                            batch_size = batch_size,

                                            shuffle = False)

    

    h = model.fit_generator(train_idg,

                            len(train_idg),

                            validation_data = test_idg,

                            validation_steps = len(test_idg),

                            shuffle = False).history

    

    train_loss.append(h['loss'][0])

    test_loss.append(h['val_loss'][0])

    train_acc.append(h['acc'][0])

    test_acc.append(h['val_acc'][0])

    

    if test_acc[-1] > best_acc:

        best_acc = test_acc[-1]

        model.save('Model.h5')

        

    alpha *= .9
print('Best accuracy %.2f' % (best_acc*100), '%', sep='')



plt.figure(figsize=(8, 6))

plt.plot(train_loss, 'b-', label='train_loss')

plt.plot(test_loss, 'r-', label='test_loss')

plt.title('Losses')

plt.legend()

plt.show()



plt.figure(figsize=(8, 6))

plt.plot(train_acc, 'b-', label='train_acc')

plt.plot(test_acc, 'r-', label='test_acc')

plt.title('Accuracies')

plt.legend()

plt.show()
test_idg = ImageDataGenerator(preprocessing_function = lambda x: x/127.5-1)

test_idg = test_idg.flow_from_directory(test_dir,

                                        (size, size),

                                        class_mode = 'binary',

                                        batch_size = batch_size,

                                        shuffle = False)



y_pred = np.round(model.predict_generator(test_idg, len(test_idg)))

cm = confusion_matrix(test_idg.classes, y_pred)



fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...

ax.set(xticks = np.arange(cm.shape[1]),

       yticks = np.arange(cm.shape[0]),

       # ... and label them with the respective list entries

       xticklabels = ['Man', 'Woman'], yticklabels = ['Man', 'Woman'],

       title = 'Test Confusion Matrix',

       ylabel = 'True label',

       xlabel = 'Predicted label')



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

fmt = 'd'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, format(cm[i, j], fmt),

                ha="center", va="center",

                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()

plt.show()
valid_idg = ImageDataGenerator(preprocessing_function = lambda x: x/127.5-1)

valid_idg = valid_idg.flow_from_directory(valid_dir,

                                         (size, size),

                                         class_mode = 'binary',

                                         batch_size = batch_size,

                                         shuffle = False)



y_pred = np.round(model.predict_generator(valid_idg, len(valid_idg)))

cm = confusion_matrix(valid_idg.classes, y_pred)



fig, ax = plt.subplots()

im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

ax.figure.colorbar(im, ax=ax)

# We want to show all ticks...

ax.set(xticks = np.arange(cm.shape[1]),

       yticks = np.arange(cm.shape[0]),

       # ... and label them with the respective list entries

       xticklabels = ['Man', 'Woman'], yticklabels = ['Man', 'Woman'],

       title = 'Valid Confusion Matrix',

       ylabel = 'True label',

       xlabel = 'Predicted label')



# Rotate the tick labels and set their alignment.

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")



# Loop over data dimensions and create text annotations.

fmt = 'd'

thresh = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, format(cm[i, j], fmt),

                ha="center", va="center",

                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()

plt.show()