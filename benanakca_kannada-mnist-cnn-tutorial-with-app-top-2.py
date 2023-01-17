import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization



from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D,LeakyReLU

from tensorflow.keras.optimizers import RMSprop,Nadam,Adadelta

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.regularizers import l2



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
tf.test.gpu_device_name()
raw_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

raw_test = pd.read_csv('../input/Kannada-MNIST/test.csv')
raw_train.iloc[[0,-1],[1,-1]] # First and last values of dataset
num = raw_train.label.value_counts()

sns.barplot(num.index,num)

numbers = num.index.values
num=6

number = raw_train.iloc[num,1:].values.reshape(28,28)

print("Picture of "+ str(num) + "in Kannada style")

plt.imshow(number, cmap=plt.get_cmap('gray'))

plt.show()

x = raw_train.iloc[:, 1:].values.astype('float32') / 255

y = raw_train.iloc[:, 0] # labels
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=42) 
x_train.shape
x_train = x_train.reshape(-1, 28, 28,1)

x_val = x_val.reshape(-1, 28, 28,1)

y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(28, 28, 1)),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(64,  (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(128, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),    

    

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    tf.keras.layers.LeakyReLU(alpha=0.1),

    tf.keras.layers.Conv2D(256, (3,3), padding='same'),

    tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),##

    tf.keras.layers.LeakyReLU(alpha=0.1),



    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256),

    tf.keras.layers.LeakyReLU(alpha=0.1),

 

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(10, activation='softmax')

])
optimizer = RMSprop(learning_rate=0.002,###########optimizer = RMSprop(learning_rate=0.0025,###########

    rho=0.9,

    momentum=0.1,

    epsilon=1e-07,

    centered=True,

    name='RMSprop')

model.compile(loss='categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
batch_size = 1024

num_classes = 10

epochs = 40

# An observation code for our dataset

datagen_try = ImageDataGenerator(rotation_range=15,

                             width_shift_range = 0.15,

                             height_shift_range = 0.15,

                             shear_range = 0.15,

                             zoom_range = 0.4,)

# fit parameters from data

datagen_try.fit(x_train)

# configure batch size and retrieve one batch of images

for x_batch, y_batch in datagen_try.flow(x_train, y_train, batch_size=9):

	# create a grid of 3x3 images

	for i in range(0, 9):

		plt.subplot(330 + 1 + i)

		plt.imshow(x_batch[i].reshape(28, 28), cmap=plt.get_cmap('gray'))

	# show the plot

	plt.show()

	break
datagen_train = ImageDataGenerator(rotation_range = 10,

                                   width_shift_range = 0.25,

                                   height_shift_range = 0.25,

                                   shear_range = 0.1,

                                   zoom_range = 0.4,

                                   horizontal_flip = False)



datagen_val = ImageDataGenerator() 





step_train = x_train.shape[0] // batch_size

step_val = x_val.shape[0] // batch_size



learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau( 

    monitor='loss',    # Quantity to be monitored.

    factor=0.25,       # Factor by which the learning rate will be reduced. new_lr = lr * factor

    patience=2,        # The number of epochs with no improvement after which learning rate will be reduced.

    verbose=1,         # 0: quiet - 1: update messages.

    mode="auto",       # {auto, min, max}. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; 

                       # in the max mode it will be reduced when the quantity monitored has stopped increasing; 

                       # in auto mode, the direction is automatically inferred from the name of the monitored quantity.

    min_delta=0.0001,  # threshold for measuring the new optimum, to only focus on significant changes.

    cooldown=0,        # number of epochs to wait before resuming normal operation after learning rate (lr) has been reduced.

    min_lr=0.00001     # lower bound on the learning rate.

    )



es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)
history = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),

                              steps_per_epoch=len(x_train)//batch_size,

                              epochs=epochs,

                              validation_data=(x_val, y_val),

                              validation_steps=50,

                              callbacks=[learning_rate_reduction, es],

                              verbose=2)
tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
model.evaluate(x_val, y_val, verbose=2);
y_predicted = model.predict(x_val)

y_grand_truth = y_val

y_predicted = np.argmax(y_predicted,axis=1)

y_grand_truth = np.argmax(y_grand_truth,axis=1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_grand_truth, y_predicted)
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cm,fmt=".0f", annot=True,linewidths=0.1, linecolor="purple", ax=ax)

plt.xlabel("Predicted")

plt.ylabel("Grand Truth")

plt.show()
scores = np.zeros((10,3))

def calc_F1(num):

  TP = cm[num,num]

  FN = np.sum(cm[num,:])-cm[num,num]

  FP = np.sum(cm[:,num])-cm[num,num]

  precision = TP/(TP+FP)

  recall = TP/(TP+FN)

  F1_score = 2*(recall * precision) / (recall + precision)

  return precision, recall, F1_score

for i in range(10):

   precision, recall, F1_score = calc_F1(i)

   scores[i,:] = precision, recall, F1_score

scores_frame = pd.DataFrame(scores,columns=["Precision", "Recall", "F1 Score"], index=[list(range(0, 10))])
f, ax = plt.subplots(figsize = (4,6))

ax.set_title('Number Scores')

sns.heatmap(scores_frame, annot=True, fmt=".3f", linewidths=0.5, cmap="PuBu", cbar=True, ax=ax)

bottom, top = ax.get_ylim()

plt.ylabel("")

ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()
raw_dig = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

raw_dig.head()

x_dig = raw_dig.iloc[:, 1:].values.astype('float32') / 255

y_dig = raw_dig.iloc[:, 0].values



x_dig = x_dig.reshape(-1,28,28,1)

y_dig = to_categorical(y_dig)

model.evaluate(x_dig, y_dig, verbose=2)
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

raw_test_id=raw_test.id

raw_test=raw_test.drop("id",axis="columns")

raw_test=raw_test / 255

test=raw_test.values.reshape(-1,28,28,1)

test.shape
sub=model.predict(test)     ##making prediction

sub=np.argmax(sub,axis=1) ##changing the prediction intro labels



sample_sub['label']=sub

sample_sub.to_csv('submission.csv',index=False)