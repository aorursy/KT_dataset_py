import tensorflow as tf

print(tf.__version__)
import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.models import Model
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()



#Split them into train & test

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = y_train.flatten(), y_test.flatten()
print("x_train.shape:", x_train.shape)

print("y_train.shape", y_train.shape)
# number of classes

K = len(set(y_train))

print("number of classes:", K)
i = Input(shape=x_train[0].shape)



x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)

x = BatchNormalization()(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)



x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D((2, 2))(x)



x = Flatten()(x)

x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)



model = Model(i, x)
# Note: make sure you are using the GPU for this.

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50)
batch_size = 32

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)

steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=steps_per_epoch, epochs=50)
# Plot loss per iteration

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10,5]

plt.plot(r.history['loss'], label='loss')

plt.plot(r.history['val_loss'], label='val_loss')

plt.legend()
# Plot accuracy per iteration

plt.plot(r.history['accuracy'], label='acc')

plt.plot(r.history['val_accuracy'], label='val_acc')

plt.legend()
# Plot confusion matrix

from sklearn.metrics import confusion_matrix

import itertools

plt.rcParams['figure.figsize'] = [10,7]



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):



  if normalize:

      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

      print("Normalized confusion matrix")

  else:

      print('Confusion matrix, without normalization')



  print(cm)



  plt.imshow(cm, interpolation='nearest', cmap=cmap)

  plt.title(title)

  plt.colorbar()

  tick_marks = np.arange(len(classes))

  plt.xticks(tick_marks, classes, rotation=45)

  plt.yticks(tick_marks, classes)



  fmt = '.2f' if normalize else 'd'

  thresh = cm.max() / 2.

  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

      plt.text(j, i, format(cm[i, j], fmt),

               horizontalalignment="center",

               color="white" if cm[i, j] > thresh else "black")



  plt.tight_layout()

  plt.ylabel('True label')

  plt.xlabel('Predicted label')

  plt.show()





p_test = model.predict(x_test).argmax(axis=1)

cm = confusion_matrix(y_test, p_test)

plot_confusion_matrix(cm, list(range(10)))
labels = '''airplane

automobile

bird

cat

deer

dog

frog

horse

ship

truck'''.split()
misclassified_idx = np.where(p_test == y_test)[0]

i = np.random.choice(misclassified_idx)

plt.imshow(x_test[i], cmap='gray')

plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));
misclassified_idx = np.where(p_test != y_test)[0]

i = np.random.choice(misclassified_idx)

plt.imshow(x_test[i], cmap='gray')

plt.title("True label: %s Predicted: %s" % (labels[y_test[i]], labels[p_test[i]]));
# Now that the model is so large, it's useful to summarize it

model.summary()