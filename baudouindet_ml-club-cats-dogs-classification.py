import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import gc

import os

import pickle



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical

from keras.optimizers import RMSprop

from keras.models import Sequential

from keras import preprocessing

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
cats_paths = []

cats_path = '/kaggle/input/cat-and-dog/training_set/training_set/cats'

for path in os.listdir(cats_path):

    if '.jpg' in path:

        cats_paths.append(os.path.join(cats_path, path))



dogs_paths = []

dogs_path = '/kaggle/input/cat-and-dog/training_set/training_set/dogs'

for path in os.listdir(dogs_path):

    if '.jpg' in path:

        dogs_paths.append(os.path.join(dogs_path, path))



len(cats_paths)
# Load data

data = np.zeros((8000, 150, 150, 3), dtype='float32')

for i in range(8000):

    if i < 4000:

        path = dogs_paths[i]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        data[i] = preprocessing.image.img_to_array(img)

    else:

        path = cats_paths[i - 4000]

        img = preprocessing.image.load_img(path, target_size=(150, 150))

        data[i] = preprocessing.image.img_to_array(img)
# Normalisation

data = data/255
# Target vector

Y = np.concatenate((np.zeros(4000), np.ones(4000)))

Y = to_categorical(Y, num_classes = 2)

print(Y)
X_train, X_cv, Y_train, Y_cv = train_test_split(data, Y, test_size = 0.1, random_state=1234)
plt.imshow(X_train[0])

print(np.argmax(Y_train[0]))
datagen = ImageDataGenerator(rotation_range = 10,

                             zoom_range = 0.2,

                             width_shift_range = 0.2,

                             height_shift_range = 0.2,

                             shear_range = 0.2,

                             horizontal_flip = True,

                             vertical_flip = False)
model = Sequential()





# 2x couches de convolution + Max pooling layer

model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (150,150,3)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())



# model.add(Conv2D(filters = 32, kernel_size = 3, strides = 2, activation ='relu'))

model.add(MaxPool2D((2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



# 2x couches de convolution + Max pooling layer

model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())



# model.add(Conv2D(filters = 64, kernel_size = 3, strides = 2, activation ='relu'))

model.add(MaxPool2D((2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))



# 2x couches de convolution + Max pooling layer

model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())



# model.add(Conv2D(filters = 64, kernel_size = 3, strides = 2, activation ='relu'))

model.add(MaxPool2D((2, 2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))





# Couche 2D --> 1D + Fully-connected + Dropout + Fully-connected

model.add(Flatten())

model.add(Dense(1024, activation = "relu"))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Dense(2, activation = "softmax"))



# Résumé de notre modèle

model.summary()
# Compile the model

model.compile(optimizer = "adam" , loss = "categorical_crossentropy", metrics=["accuracy"])



# Set a learning rate annealer

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=64), validation_data = (X_cv,Y_cv),

                              epochs = 50,

                              steps_per_epoch = X_train.shape[0]//64,

                              verbose = True,

                              callbacks=[annealer])



with open('model.pickle', 'wb') as file:

    pickle.dump(model, file)
# Check model performance

print('Validation accuracy = ' + str(round(max(history.history['val_accuracy']), 4)))



plt.figure(1, figsize=(10,5))

plt.plot(history.history['loss'], color='blue', label='training loss')

plt.plot(history.history['val_loss'], color='red', label='validation loss')

plt.legend()

plt.grid()



plt.figure(2, figsize=(10,5))

plt.plot(history.history['accuracy'], color='blue', label='training accuracy')

plt.plot(history.history['val_accuracy'], color='red', label='validation accuracy')

plt.legend()

plt.grid()
# On arrondit à l'entier le plus proche

predictions = np.argmax(model.predict(X_cv), axis=1)



confusion = confusion_matrix(np.argmax(Y_cv, axis=1), predictions)

plt.figure(1, figsize=(10,7))

sns.heatmap(confusion, annot=True, cmap='PuBu')

plt.xlabel('Predicted label')

plt.ylabel('True label')

# Définition d'une dataframe avec les predictions et vrais labels

df_predictions = pd.DataFrame({'prediction' : predictions,

                               'actual': np.argmax(Y_cv, axis=1)})



# On ajoute la colonne 'correct_pred' qui nous dit si la prédiction est bonne

df_predictions['correct_pred'] = np.where(df_predictions.prediction == df_predictions.actual, 1, 0)



# on crée un array de tous les index avec erreur sur la validation

error_index = df_predictions.loc[df_predictions.correct_pred == 0, :].index



# Maintenant on peut plot les chiffres où on s'est trompé

plt.figure(1, figsize=(15,15))

for i in range(16):

    plt.subplot(4, 4, i+1)

    plt.imshow(X_cv[error_index[i],:,:,:])

    plt.title('actual : ' + str(df_predictions.loc[error_index[i]].actual) + ' - predicted : ' + str(df_predictions.loc[error_index[i]].prediction))