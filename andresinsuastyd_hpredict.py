import tensorflow as tf

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

from scipy import optimize

from tensorflow import keras

from tensorflow.keras import layers

print('TensorFlow Version:',tf.__version__)
INPATH = '/kaggle/input/waterfilling-lte/h/'

INPATH_R = '/kaggle/input/waterfilling-lte/rho/'

imgurls = os.listdir(INPATH)

outurls = os.listdir(INPATH_R)

print("cantidad de imagenes H:",len(imgurls))

print("cantidad de imagenes rho:",len(outurls))
def load_image(filename):

    inimg = tf.cast(tf.io.decode_image(tf.io.read_file(INPATH + '/' + filename)),tf.float32)

    inimg = (inimg/127.5)-1 # valores entre -1 y 1

    return inimg
def load_output(filename):

    inimg = tf.cast(tf.io.decode_image(tf.io.read_file(INPATH_R + '/' + filename)),tf.float32)

    inimg = (inimg/127.5)-1 # valores entre -1 y 1

    return inimg
img_prueba = load_image(imgurls[0])

plt.imshow((np.squeeze(img_prueba+1)/2))

plt.show()

print(img_prueba.shape)
def creando_dato(imagen):

    data1 = np.array(imagen[:,0:20,0])

    data2 = np.array(imagen[:,20:40,0])

    data3 = np.array(imagen[:,40:60,0])

    data4 = np.array(imagen[:,60:80,0])

    data5 = np.array(imagen[:,80:100,0])

    data6 = np.array(imagen[:,100:120,0])

    data7 = np.array(imagen[:,120:140,0])

    data8 = np.array(imagen[:,140:160,0])

    data9 = np.array(imagen[:,160:180,0])

    data10 = np.array(imagen[:,180:200,0])

    return np.concatenate((data1,data2,data3,data4,data5,data6,data7,data8,data9,data10))
data = creando_dato(load_image(imgurls[900]))

for i in range(200):

    indice = i

    inp = creando_dato(load_image(imgurls[indice]))

    data = np.concatenate((data, inp))

print("tamaño:",data.shape)



    

    
def build_model():

  model = keras.Sequential([

    layers.Dense(10, activation='relu'),

    #layers.Dense(512, activation='relu'),

#     layers.Dense(50,activation='relu'),

      

#     layers.Dense(25,activation='relu'),

#     layers.Dense(12,activation='relu'),

#     layers.Dense(6,activation='relu'),

#     layers.Dense(3,activation='relu'),      

    layers.Dense(10,activation='tanh')

  ])



  optimizer = tf.keras.optimizers.RMSprop(0.001)



  model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

  return model
def plot_history(history):

  hist = pd.DataFrame(history.history)

  hist['epoch'] = history.epoch



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Abs Error')

  plt.plot(hist['epoch'], hist['mae'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mae'],

           label = 'Val Error')



  plt.legend()



  plt.figure()

  plt.xlabel('Epoch')

  plt.ylabel('Mean Square Error')

  plt.plot(hist['epoch'], hist['mse'],

           label='Train Error')

  plt.plot(hist['epoch'], hist['val_mse'],

           label = 'Val Error')

  plt.legend()

  plt.show()

class PrintDot(keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs):

    if epoch % 100 == 0: print('')

    print('.', end='')
x=data[:,:10]

y=data[:,10:]

print("Data tamaño: ", data.shape)

print("Entrada tamaño: ",x.shape)

print("Salida Tamaño: ", y.shape)
model = build_model()



# The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



history = model.fit(x, y, epochs=100, batch_size=10000,

                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

                    #validation_split = 0.2, verbose=0, callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.tail()



plot_history(history)
img_prueba = load_image(imgurls[900])

data_test=creando_dato(img_prueba)[0:100,:]

y=model.predict(data_test[:,:10])

print("data Test: ", data_test.shape)
plt.figure()

plt.subplot(121)

plt.plot(data_test[:,10],'b',label='Ground True')

plt.plot(y[:,0],'r',label='Predicted')

plt.title('Closer Prediction')

plt.xlabel('Subcarrier')

plt.ylabel('$|h_i|^2_{normalized}$')

plt.legend()

# plt.ylim([-0.96,-0.5])



plt.subplot(122)

plt.plot(data_test[:,-1],'b',label='Ground True')

plt.plot(y[:,-1],'r',label='Predicted')

plt.title('Last Prediction')

plt.xlabel('Subcarrier')

plt.ylabel('$|h_i|^2_{normalized}$')

# plt.ylim([-0.96,-0.5])

plt.legend()

plt.tight_layout()



plt.show()
plt.figure(figsize=(10,10))

plt.subplot(131)

plt.imshow((data_test[:,:10]+1)/2)

plt.title('INPUT')

plt.xlabel('Time (ms)')

plt.ylabel('$|h_i|^2$')



plt.subplot(132)

plt.imshow((data_test[:,10:]+1)/2)

plt.title('Ground True')

plt.xlabel('Time (ms)')

plt.ylabel('$|h_i|^2$')



plt.subplot(133)

plt.imshow((y+1)/2)
from sklearn.metrics import r2_score

r2_score(data_test[:,10:], y)
test = creando_dato(load_image(imgurls[200]))

for i in range(200,400):

    indice = i

    inp = creando_dato(load_image(imgurls[indice]))

    test = np.concatenate((test, inp))

print("tamaño:",data.shape)
y_test = model.predict(test[:,:10])
r2=r2_score(test[:,10:], y_test)

print('El valor de R2 para datos de prueba es:',r2)
def waterfilling(h,Pmax=1000):

    h=((h+1)/2)*11.769 # recibe coeficientes normalizados entre 0 y 1, se reescala para tener valores originales

    h=np.transpose(h)

    k=512



    def f(u):

        return (np.sum(np.maximum(np.zeros(k),1/u-1/(h)))-Pmax)**2



    u_opt=optimize.fminbound(f, 0, np.amax(h), xtol=1e-10)

    rho=np.zeros(k)



    for i in range(k):

        if u_opt < h[0,i]:

            rho[i]=1/u_opt-1/h[0,i]

        else:

            rho[i]=0

    return rho
rho_test=waterfilling(y_test[0:512,:1])

plt.plot((y_test[0:512,0:1]+1)/2*11.769,label="$|h_i|^2$")

plt.plot(rho_test,label="$rho_i$")

plt.legend()

plt.show()