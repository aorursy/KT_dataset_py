import tensorflow as tf



config = tf.ConfigProto()

config.gpu_options.allow_growth = True

tf.keras.backend.set_session(tf.Session(config=config))



from keras import backend as K



# from keras.applications.vgg16 import VGG16

# from keras.applications.vgg16 import preprocess_input



from keras.models import Sequential

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam, RMSprop

from keras.callbacks import LearningRateScheduler

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

# from sklearn.metrics import accuracy_score, classification_report

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import LeakyReLU

from keras.layers.normalization import BatchNormalization
train      = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test       = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sample_sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

dig        = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
print("Train set shape = " +str(train.shape))

print("Test set shape = " +str(test.shape))

print("Sub set shape = " +str(sample_sub.shape))

print("Dig set shape = " +str(dig.shape))
# Dados de TREINAMENTO  (todas as linhas, da segunda coluna em diante)

x_train = train.values[:,1:].reshape(train.shape[0], 28, 28, 1)

# Transformando os valores do CSV em imagem (float32)

x_train = x_train.astype('float32')

# Deixando os píxels na mesma escala, ie. deixando em uma escala de 0 a 1 para não gerar viés

x_train = x_train / 255.0

# Separando os labels de treinamento    

y_train = train.values[:,0]

# Separando dados de treinamento e validação

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.1, random_state=2019)

# Dados de TESTE (todas as linhas, da segunda coluna em diante)

x_test = test.values[:,1:].reshape(test.shape[0], 28, 28, 1)

# Transformando os valores do CSV em imagem (float32)

x_test = x_test.astype('float32')

# Deixando os píxels na mesma escala, ie. deixando em uma escala de 0 a 1 para não gerar viés

x_test = x_test / 255.0
# Transformando as classes numéricas (0,1,2,3,4,5,6,7,8,9) em binários ([1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0],...,[0,0,0,0,0,0,0,0,0,1])

lb = preprocessing.LabelBinarizer()

y_train = lb.fit_transform(y_train)

y_valid = lb.fit_transform(y_valid)
modelo = Sequential()



modelo.add(Conv2D(64,  (3,3), padding='same', input_shape=(28, 28, 1)))

modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

modelo.add(LeakyReLU(alpha=0.1))



modelo.add(MaxPooling2D(2, 2))

modelo.add(Dropout(0.2))



modelo.add(Conv2D(128, (3,3), padding='same'))

modelo.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

modelo.add(LeakyReLU(alpha=0.1))

modelo.add(Conv2D(128, (3,3), padding='same'))

modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

modelo.add(LeakyReLU(alpha=0.1))



modelo.add(MaxPooling2D(2,2))

modelo.add(Dropout(0.2))



modelo.add(Conv2D(256, (3,3), padding='same'))

modelo.add(BatchNormalization(momentum=0.2, epsilon=1e-5, gamma_initializer="uniform"))

modelo.add(LeakyReLU(alpha=0.1))

modelo.add(Conv2D(256, (3,3), padding='same'))

modelo.add(BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer="uniform"))

modelo.add(LeakyReLU(alpha=0.1))



modelo.add(MaxPooling2D(2,2))

modelo.add(Dropout(0.2))



modelo.add(Flatten())

modelo.add(Dense(256))

modelo.add(LeakyReLU(alpha=0.1))



modelo.add(BatchNormalization())

modelo.add(Dense(10, activation='softmax'))
initial_learningrate=2e-3

def lr_decay(epoch):#lrv

    return initial_learningrate * 0.99 ** epoch

modelo.compile(loss= 'categorical_crossentropy', optimizer= RMSprop(lr=initial_learningrate) , metrics=['accuracy'])

modelo.summary()
epochs=100

batch_size=200

history = modelo.fit(x_train, y_train,

          epochs=epochs,

          batch_size= batch_size,

          callbacks=[LearningRateScheduler(lr_decay)],

          validation_data=(x_valid, y_valid))
H = history

plt.style.use("bmh")

plt.figure()

plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")

plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")

plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy on Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="best")
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='best')

plt.show()
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='best')

plt.show()
scores = modelo.predict(x_test)

score_classes = np.argmax(scores, axis = 1)

x_dig=dig.drop('label',axis=1).iloc[:,:].values

print(x_dig.shape)

x_dig = x_dig.reshape(x_dig.shape[0], 28, 28,1)

print(x_dig.shape)

y_dig=dig.label

print(y_dig.shape)

preds_dig=modelo.predict_classes(x_dig/255)

metrics.accuracy_score(preds_dig, y_dig)
output = pd.DataFrame({'id': sample_sub['id'],

                       'label': score_classes})

output.head()
output.to_csv('submission.csv', index=False)