import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Puro = np.load('../input/train_images_pure.npy')
plt.subplot(141)
plt.imshow(Puro[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(Puro[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(Puro[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(Puro[3], cmap=plt.get_cmap('gray'))
plt.show()
Rot = np.load('../input/train_images_rotated.npy')
plt.subplot(141)
plt.imshow(Rot[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(Rot[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(Rot[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(Rot[3], cmap=plt.get_cmap('gray'))
plt.show()
Ruido = np.load('../input/train_images_noisy.npy')
plt.subplot(141)
plt.imshow(Ruido[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(Ruido[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(Ruido[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(Ruido[3], cmap=plt.get_cmap('gray'))
plt.show()
Tudo = np.load('../input/train_images_both.npy')
plt.subplot(141)
plt.imshow(Tudo[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(Tudo[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(Tudo[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(Tudo[3], cmap=plt.get_cmap('gray'))
plt.show()
Teste = np.load('../input/Test_images.npy')
plt.subplot(141)
plt.imshow(Teste[0], cmap=plt.get_cmap('gray'))
plt.subplot(142)
plt.imshow(Teste[1], cmap=plt.get_cmap('gray'))
plt.subplot(143)
plt.imshow(Teste[2], cmap=plt.get_cmap('gray'))
plt.subplot(144)
plt.imshow(Teste[3], cmap=plt.get_cmap('gray'))
plt.show()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
semente = 7
np.random.seed(semente)
Y = pd.read_csv("../input/train_labels.csv",index_col=0)
Xfit = Puro.reshape(Rot.shape[0], 1, 28, 28).astype('float32')
Xteste = Rot.reshape(Rot.shape[0], 1, 28, 28).astype('float32')
Xteste2 = Tudo.reshape(Rot.shape[0], 1, 28, 28).astype('float32')
Xfit = Xfit / 255
Xteste = Xteste / 255
Xteste2 = Xteste2 / 255

Yfit = np_utils.to_categorical(Y)
num_classes = Yfit.shape[1]

Xtreino,Xvalid,Ytreino,Yvalid = train_test_split(Xfit,Yfit, test_size = 0.2)
def baseline_modelo1():
    modelo1 = Sequential()
    modelo1.add(Conv2D(32, (5,5), input_shape=(1, 28, 28), activation='relu'))
    modelo1.add(Dropout(0.2))
    modelo1.add(Flatten())
    modelo1.add(Dense(128, activation='relu'))
    modelo1.add(Dense(num_classes, activation='softmax'))
    modelo1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelo1
# build the model
modelo1 = baseline_modelo1()
modelo1.summary()
callbacks = [EarlyStopping(monitor = 'val_loss', patience = 2)]
modelo1.fit(Xtreino, Ytreino, validation_data=(Xvalid,Yvalid), epochs=20, 
          batch_size=400, verbose=1, callbacks = callbacks)
#Foi utilizado epochs=20
scores = modelo1.evaluate(Xteste, Yfit, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
scores = modelo1.evaluate(Xteste2, Yfit, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
from keras.layers.convolutional import MaxPooling2D
def baseline_modelo2():
    modelo2 = Sequential()
    modelo2.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    modelo2.add(MaxPooling2D(pool_size=(2, 2)))
    modelo2.add(Dropout(0.2))
    modelo2.add(Flatten())
    modelo2.add(Dense(128, activation='relu'))
    modelo2.add(Dense(num_classes, activation='softmax'))
    modelo2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelo2
modelo2 = baseline_modelo2()
modelo2.summary()
modelo2.fit(Xtreino, Ytreino, validation_data=(Xvalid,Yvalid), epochs=20, 
          batch_size=400, verbose=1, callbacks = callbacks)
#Foi utilizado epochs=20
scores = modelo2.evaluate(Xteste, Yfit, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
scores = modelo2.evaluate(Xteste2, Yfit, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
Xnovo = Tudo.reshape(Rot.shape[0], 1, 28, 28).astype('float32')
Xnovo = Xnovo/255

Ynovo = np_utils.to_categorical(Y)

Xt,Xv,Yt,Yv = train_test_split(Xnovo,Ynovo, test_size = 0.2)
Teste = np.load('../input/Test_images.npy')
Teste = Teste.reshape(Teste.shape[0], 1, 28, 28).astype('float32')
def modelo():
    modelo = Sequential()
    modelo.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Conv2D(15, (3, 3), activation='relu'))
    modelo.add(MaxPooling2D(pool_size=(2, 2)))
    modelo.add(Dropout(0.2))
    modelo.add(Flatten())
    modelo.add(Dense(128, activation='relu'))
    modelo.add(Dense(50, activation='relu'))
    modelo.add(Dense(num_classes, activation='softmax'))
    modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return modelo
modelo = modelo()
modelo.summary()
modelo.fit(Xt, Yt, validation_data=(Xv,Yv), epochs=20, 
          batch_size=200, verbose=1, callbacks = callbacks)
#Foi utilizado epochs=20
def encode(p):
    resp=[]
    for linha in p:
        flag=False
        for j in range(len(linha)):
            if linha[j]==1:
                resp.append(j)
                flag=True
        if not flag:
            resp.append(8)
    return resp
Pred1 = modelo1.predict(Teste)

Pred2 = modelo2.predict(Teste)
Pred = modelo.predict(Teste)
P1 = pd.DataFrame(columns = ['Id','label'])
P1.label = encode(Pred1)
P1.Id = range(len(Teste))
P1.to_csv("Pred1.csv",index=False)

P2 = pd.DataFrame(columns = ['Id','label'])
P2.label = encode(Pred2)
P2.Id = range(len(Teste))
P2.to_csv("Pred2.csv",index=False)

P = pd.DataFrame(columns = ['Id','label'])
P.label = encode(Pred)
P.Id = range(len(Teste))
P.to_csv("Pred.csv",index=False)


