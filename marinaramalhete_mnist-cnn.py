# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical      # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
np.random.seed(2)
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head()
train.shape
test.shape
# Preparando os dados



Y_train = train['label']

X_train = train.drop(labels = ['label'], axis = 1)



g = sns.countplot(Y_train)

Y_train.value_counts()
# Tratando os valores nulos



X_train.isnull().any().describe()
test.isnull().any().describe()
# Normalizacao

# Como em uma imagem em escala de cinza o valor de cada pixel é a única amostra do espaço de cores, esse valor irá variar no intervalo [0, 255]



X_train = X_train/255.0

test = test/255.0
# Reshape em 3 dimensões (altura = 28px, largura = 28px, canal = 1)

# Cada imagem do dataset possui 28 X 28 pixels, com os valores dos pixels em escala de cinza

# É preciso fazer essa redimensao pois o keras requer uma dimensao extra no final



X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
# Label encoding

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

# Os rótulos são números de 10 dígitos de 0 a 9



Y_train = to_categorical(Y_train, num_classes = 10)



# to_categorical -> Converte um vetor de classe (inteiros) em matriz de classe binária.
# Split conjunto de treinamento e validacao



random_seed = 2



# Split the train and the validation set for the fitting

X_train, X_val, Y_train, Y_val = train_test_split(X_train,

                                                  Y_train,

                                                  test_size = 0.1,

                                                  random_state = random_seed)
g = plt.imshow(X_train[0][:,:,0])
# CNN (construindo a rede neural)

# Set CNN model

# Arquitetura (tutorial) ->  [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



# Inicia a rede neural

model = Sequential()



# Uso de 32 features e o array tera o formato de 5x5

# MaxPooling (agrupamento pooling) para reduzir o tamanho do mapa de features

# Conv2D -> segunda camada de convolucao (torna a rede mais profunda)

# Flatten -> converte a estrutura de dados 2D resultado da camada anterior em 1D

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu', input_shape = (28, 28, 1)))

model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3, 3) ,padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))

model.add(Dropout(0.25))



# Conectamos todas as camadas

# relu -> funcao de ativacao "retificadora"

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# Optimizer

# Esta função melhora iterativamente os parâmetros (filtra os valores do núcleo, pesos e polarização dos neurônios ...), a fim de minimizar a perda

# A atualização RMSProp ajusta o método Adagrad de uma maneira muito simples, na tentativa de reduzir sua agressividade



# Define o optimizer

optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay = 0.0)



# Compila o model

# Metrica: acuracia -> maior preocupacao no treinamento deste tipo de modelo (duvida ainda)

model.compile(optimizer = optimizer,

              loss = "categorical_crossentropy",

              metrics = ["accuracy"])
# Set a learning rate annealer



learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', 

                                            patience = 3, 

                                            verbose = 1, 

                                            factor = 0.5, 

                                            min_lr = 0.00001)
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
# Aumentando os dados



# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(

        featurewise_center = False,  # set input mean to 0 over the dataset

        samplewise_center = False,  # set each sample mean to 0

        featurewise_std_normalization = False,  # divide inputs by std of the dataset

        samplewise_std_normalization = False,  # divide each input by its std

        zca_whitening = False,  # apply ZCA whitening

        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip = False,  # randomly flip images

        vertical_flip = False)  # randomly flip images

datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch = X_train.shape[0] // batch_size,

                              callbacks = [learning_rate_reduction])
# Matriz de confusao



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)