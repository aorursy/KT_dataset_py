# Importando bibliotecas, bases de dado e ferramentas relevantes

%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

pure = np.load('../input/train_images_pure.npy')
rotated = np.load('../input/train_images_rotated.npy')
noisy = np.load('../input/train_images_noisy.npy')
both = np.load('../input/train_images_both.npy')

test = np.load('../input/Test_images.npy')
#Visualizando uma imagem equivalente de cada porcao

plt.subplot(141)
plt.imshow(pure[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(rotated[0], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(noisy[0], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(both[0], cmap=plt.get_cmap('gray'))
plt.show()
#Vimos como sao as imagens de cada porcao da base de treino, agora veremos a de teste
plt.imshow(test[0], cmap=plt.get_cmap('gray'))
numPixels = noisy.shape[1] * noisy.shape[2]
datagen = ImageDataGenerator(rotation_range=10)
X_train = noisy.reshape(noisy.shape[0], 28, 28, 1).astype('float32')
X_train = X_train / 255.0 #Normalizing

Y_train = pd.read_csv('../input/train_labels.csv').drop(columns=['Id'])
Y_train = to_categorical(Y_train)
#CONSTRUINDO AS CNNs
numero = 3
model = [0]*numero

for j in range(numero):
    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 3, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Dropout(0.4))

    model[j].add(Conv2D(128, kernel_size = 4, activation='relu'))
    model[j].add(BatchNormalization())
    model[j].add(Flatten())
    model[j].add(Dropout(0.4))
    model[j].add(Dense(10, activation='softmax'))

    # COMPILADO COM O OTIMIZADOR DE ADAM
    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x) # Para diminuir a taxa do treinamento
epochs = 45
from sklearn.model_selection import train_test_split
#TREINANDO
for j in range (numero):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
X_test = test.reshape(test.shape[0], 28, 28, 1).astype('float32')
X_test = X_test / 255.0 #Normalizing

#FAZ A PREDICAO E SELECIONA O RESULTADO MAIS EXPRESSIVO
results = np.zeros( (X_test.shape[0],10) ) 
for j in range(numero):
    results = results + model[j].predict(X_test)
results = np.argmax(results,axis = 1)
Y_test = pd.Series(results,name="label")

submission = pd.concat([pd.Series(range(1,test.shape[0]),name = "Id"),Y_test],axis = 1)

Y_test.to_csv("submission.csv", index=False)
plt.figure(figsize=(10,4))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_test[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("Predicao=%d" % results[i],y=0.9)
    plt.axis('off')
plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.show()