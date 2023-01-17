import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from kerastuner import HyperModel

from kerastuner.tuners import RandomSearch

from tensorflow.python.keras.callbacks import ModelCheckpoint

import seaborn as sns

from pylab import rcParams

from sklearn.metrics import confusion_matrix
# setting figure size for the whole program

rcParams['figure.figsize'] = 15, 15

NUM_CLASSES = 10

INPUT_SHAPE = (28, 28, 1)
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

X = train.iloc[:, 1:] # 784 features representing a 28*28 Black and white pictures

y = train.iloc[:, 0] # the first column is the label 
X /= 256

test /= 256

X = X.values.reshape((-1, ) + INPUT_SHAPE)

test = test.values.reshape((-1, ) + INPUT_SHAPE)
rows = 5

cols = 5

fig=plt.figure()



for i in range(rows * cols):

    idx = np.random.randint(0, X.shape[0], dtype='int')

    # this will give us next subplot

    ax = fig.add_subplot(rows, cols, i + 1)

    subplot_title=((idx, y[idx]))

    ax.set_title(subplot_title)

    ax.axis('off')

    # passing cmap for balck and white images

    plt.imshow(X[idx, :, :, 0], cmap='Greys')

plt.show()
y.value_counts().sort_index().plot(kind='bar')

plt.xticks(rotation=0)

plt.show()
# Deriving from keras's Hypermodel

class CNNHyperModel(HyperModel):

    def __init__(self, input_shape, num_classes):

        self.input_shape = input_shape

        self.num_classes = num_classes



    def build(self, hp):

        model = tf.keras.models.Sequential()

        # doing a 1-dilated convolution with 3 * 3 kernel

        # on top of a 3-dilated convolution with 3 * 3 kernel

        # will give us a receptive field of 9 * 9

        #

        #    first layer       third layer

        # o    o    o

        #

        #     x    x    x        _ _ _

        # o    o    o           |o    |

        #                   →   |     |

        #     x    x    x       |_ _ x|

        # o    o    o

        #

        #     x    x    x

        model.add(tf.keras.layers.Conv2D(8, 3, padding='same', dilation_rate=(3, 3), \

                                 activation='relu', input_shape=self.input_shape))

        model.add(tf.keras.layers.Conv2D(16, 3, padding='same', \

                                         activation='relu'))

        

        # Hyperparameter for choosing between max pooling and average pooling.

        # If max_or_avg is 2 then three-max_or_avg will be 1 

        # so we'll do avrage pooling with 2 * 2 kernal and 

        # max pooling with 1 * 1 kernel, which will do nothing.

        # Otherwise if max_or_avg is 1 then three-max_or_avg will be 2

        # and we'll do average pooling with 1 * 1, which will do nothing kernal  

        # and max pooling with 2 * 2 kernel.

        max_or_avg =hp.Int(

                   'max_or_avg',

                   min_value=1,

                   max_value=2,

                   step=1,

                   default=1)

        three = hp.Fixed('three', 3)

        model.add(tf.keras.layers.AvgPool2D(pool_size=max_or_avg, name=f'first_avg_pool_{max_or_avg}'))

        model.add(tf.keras.layers.MaxPool2D(pool_size=three - max_or_avg))

        model.add(tf.keras.layers.BatchNormalization())

        

        model.add(tf.keras.layers.Conv2D(32, 3, padding='same', dilation_rate=(3, 3), \

                                 activation='relu'))

        model.add(tf.keras.layers.Conv2D(64, 3, padding='same', \

                                         activation='relu'))

        model.add(tf.keras.layers.AvgPool2D(pool_size=max_or_avg, name=f'second_avg_pool_{max_or_avg}'))

        model.add(tf.keras.layers.MaxPool2D(pool_size=three - max_or_avg))

        model.add(tf.keras.layers.BatchNormalization())

        

        model.add(tf.keras.layers.Flatten())



        # hyperparameter for tuning dropout rate

        drop_rate = hp.Float('dropout',

                             min_value=0.3,

                             max_value=0.7,

                             default=0.3,

                             step=0.2,

                            )

        model.add(tf.keras.layers.Dropout(rate=drop_rate, 

                                          name=f'drop{drop_rate}'))

        # tuning number of units for the dense layer

        units=hp.Int('units',

                     min_value=28,

                     max_value=4*28,

                     step=28,

                     default=28

                    )

        model.add(tf.keras.layers.Dense(

                  units=units,

                  name=f'Dense{units}',

                  activation='relu'

                ))

        model.add(tf.keras.layers.Dense(self.num_classes))

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        

        optimizer = tf.keras.optimizers.Adam()

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model
hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)



tuner = RandomSearch(

        hypermodel,

        objective='val_accuracy',

        max_trials=24,

        executions_per_trial=1,

        directory='./random_search',

        project_name='MNIST'

)



tuner.search_space_summary()
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=1/7)
search_results = tuner.search(X_train_val, y_train_val,

                              epochs=10, validation_split=1/6,

                              batch_size=100)
# This gives us 10 best models

tuner.results_summary()
best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

print(best_model.evaluate(X_test, y_test))
# decresing learning rate 10 times after each 10 epochs staring with 10**(-3)

def schedule(epoch, lr):

    return 10**(-(epoch//10)-3)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)



# saving weights of the model that best performs on validation set

cb_checkpointer_val = ModelCheckpoint(filepath = '../working/best_val.hdf5',

                                      monitor = 'val_accuracy',

                                      save_best_only = True,

                                      mode = 'auto')
X_train, X_val, y_train, y_val = train_test_split(X_train_val, 

                                                  y_train_val,

                                                  test_size=1/6)
fit_history = best_model.fit(X_train, y_train, epochs=40, batch_size=100,

                             validation_data=(X_val, y_val),

                             callbacks = [lr_schedule, cb_checkpointer_val])
plt.figure(1, figsize = (15,8)) 



plt.subplot(221)  

plt.plot(fit_history.history['accuracy'])  

plt.plot(fit_history.history['val_accuracy'])  

plt.title('model accuracy')  

plt.ylabel('accuracy')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.subplot(222)  

plt.plot(fit_history.history['loss'])  

plt.plot(fit_history.history['val_loss'])  

plt.title('model loss')  

plt.ylabel('loss')  

plt.xlabel('epoch')  

plt.legend(['train', 'valid']) 



plt.show()



idx = np.argmax(fit_history.history['val_accuracy'])

print(f"train loss - {fit_history.history['loss'][idx]}")

print(f"train accuracy - {fit_history.history['accuracy'][idx]}")

print(f"validation loss - {fit_history.history['val_loss'][idx]}")

print(f"validation accuracy - {fit_history.history['val_accuracy'][idx]}")
best_model.load_weights('./best_val.hdf5')

best_model.evaluate(X_test, y_test)
cb_checkpointer = ModelCheckpoint(filepath = '../working/best.hdf5',

                                  monitor = 'accuracy',

                                  save_best_only = True,

                                  mode = 'auto')
final_fit_history = best_model.fit(X, y, epochs=40, batch_size=100,

                                   callbacks = [lr_schedule, cb_checkpointer])
np.max(final_fit_history.history['accuracy'])
best_model.load_weights('./best.hdf5')
import matplotlib.pyplot as plt



labels = sorted(y.unique())

cm = confusion_matrix(y, best_model.predict(X).argmax(axis=-1), 

                      labels=labels)

ax = plt.subplot()

sns.heatmap(cm, annot=True, ax=ax, fmt='d')



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(ticklabels=labels) 

ax.yaxis.set_ticklabels(ticklabels=labels)

plt.yticks(rotation=0)

plt.show()
best_model.evaluate(X, y)
pred = pd.DataFrame({'ImageId' : np.arange(test.shape[0]) + 1,

                     'Label': best_model.predict(test).argmax(axis=-1)})

pred.to_csv('pred.csv', index=False)

pred