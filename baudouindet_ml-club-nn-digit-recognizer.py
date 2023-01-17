import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import gc



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical

from keras.optimizers import RMSprop

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
# Load du train

train = pd.read_csv('../input/digit-recognizer/train.csv')



# On regarde les dimensions du train et les premières lignes

print('train :', train.shape)

train.head()
# On affiche donc un barplot du count de chaque chiffre

train.label.value_counts().plot.bar(figsize=(10,5))
# Définition de X (les pixels) et y (les labels à prédire)

X = train.drop(columns=['label'])

y = train['label']



# On peut alors normaliser X pour que les valeurs soient comprisent entre 0 et 1

X = X / 255.0
# Pour cela nous utilisons la fonction reshape : array.reshape(nombre_de_matrices, hauteur, largeur, chanel)

# Quand on ne connait pas une dimension on peut mettre l'arguement -1, ainsi reshape la déduit

X = X.values.reshape(-1, 28, 28, 1)
# Commencer par afficher la première image avec la fonction : plt.imshow(2D_array, cmap=gray)

# Puis lorsque c'est bon faire une loop pour en afficher 9

plt.figure(figsize=(10,10))

for i in range(1,10):

    plt.subplot(3, 3, i)

    plt.imshow(X[i][:, :, 0], cmap='gray')
# On utilise la fonction to_categorical() en précisant le nombre de classes : to_categorical(y, num_classes = 10)

Y = to_categorical(y, num_classes = 10)



del y

gc.collect()
datagen = ImageDataGenerator(rotation_range=10,

                             zoom_range = 0.10,

                             width_shift_range=0.1,

                             height_shift_range=0.1)
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size = 0.1, random_state=1234)
# Initilise notre modèle en expliquant que nous ajoutons les couches les unes après les autres

model = Sequential()



# 2x couches de convolution + Learnable pooling layer

model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu', input_shape = (28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 32, kernel_size = 5, strides = 2, padding = 'same', activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



# 2x couches de convolution + Learnable pooling layer

model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))

model.add(BatchNormalization())



model.add(Conv2D(filters = 64, kernel_size = 5, strides = 2, padding = 'same', activation ='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



# Couche 2D --> 1D + Fully-connected + Dropout + Fully-connected

model.add(Flatten())

model.add(Dense(1024, activation = "relu"))

model.add(Dropout(0.4))

model.add(Dense(10, activation = "softmax"))



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
# dans actual on a un vecteur 1D qui nous donne le bon label pour chaque image sur la validation

actual = Y_cv.argmax(axis=1)



# dans predictions on a un vecteur 1D qui nous donne le label prédit sur la validation

predictions = model.predict(X_cv).argmax(axis=1)



confusion = confusion_matrix(actual, predictions)

plt.figure(1, figsize=(10,7))

sns.heatmap(confusion, annot=True, cmap='PuBu')

plt.xlabel('Predicted label')

plt.ylabel('True label')
# Définition d'une dataframe avec les predictions et vrais labels

df_predictions = pd.DataFrame({'prediction' : predictions,

                               'actual': actual})



# On ajoute la colonne 'correct_pred' qui nous dit si la prédiction est bonne

df_predictions['correct_pred'] = np.where(df_predictions.prediction == df_predictions.actual, 1, 0)



# on crée un array de tous les index avec erreur sur la validation

error_index = df_predictions.loc[df_predictions.correct_pred == 0, :].index



# Maintenant on peut plot les chiffres où on s'est trompé

plt.figure(1, figsize=(10,10))

for i in range(9):

    plt.subplot(3, 3, i+1)

    plt.imshow(X_cv[error_index[i],:,:,0], cmap='gray')

    plt.title('actual : ' + str(df_predictions.loc[error_index[i]].actual) + ' - predicted : ' + str(df_predictions.loc[error_index[i]].prediction))
# first : 0.992

# same first : 0.990

# re-same first : 0.992

# no drop-out : 0.98

# simple : 0.98

# 5 / 5 : 0.96

# learnable pooling + aug : 0.9967

# max pooling + aug : 0.9948

# learnable pooling + aug + dense annealing : 0.9964
# Load du test

train = pd.read_csv('../input/digit-recognizer/test.csv')



# On normalise les inputs comme sur le train

X_test = train / 255.0



# Reshape

X_test = X_test.values.reshape(-1, 28, 28, 1)



predictions_test = model.predict(X_test).argmax(axis=1)



# Load fichier de submission

submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')



submission['Label'] = predictions_test

submission.to_csv('submission.csv', index=False)