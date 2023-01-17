from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random 
from sklearn.metrics import r2_score
from scipy import optimize

modelh = keras.models.load_model('/kaggle/input/lstm-model/H_LSTM.h5')
#modelh = keras.models.load_model('/kaggle/input/h-lstm/H_LSTM.h5')
modelh.summary()
h = pd.read_csv('/kaggle/input/h-coeficientes/h_test.csv',header=None)
h.head()

h=h.values
print(type(h))
print(h.shape)
def reshape_lstm(inputs):
    return np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
def predict(h):
    h = reshape_lstm(h)
    return modelh.predict(h)
hi=np.zeros([512,200])
hi[:,0:100]=h[:,0:100]
for i in range(100):
    ho=predict(hi[:,i:i+100])
    hi[:,i+100]=np.squeeze(ho)
print(hi.shape)
plt.subplot(121)
plt.imshow(hi)
plt.subplot(122)
plt.imshow(h)
plt.show()
a=102
plt.plot(h[:,a],label='real')
plt.plot(hi[:,a],label='Estimada')
plt.legend()
plt.show()
np.savetxt("hi.csv", hi, delimiter=",")
def waterfilling(h,Pmax):
  k=512
  h=h**2
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

def waterfilling_matrix(h,Pmax):
    rho = np.zeros([h.shape[1],h.shape[0]])
    h = np.transpose(h)
    for i in range(h.shape[0]):
        rho[i,...]=waterfilling(h[i:i+1,...],Pmax)
    return np.transpose(rho)
Pmax=1000
rho2 =  waterfilling_matrix(hi,Pmax)
rho = waterfilling_matrix(h,Pmax)

plt.figure()
plt.subplot(121)
plt.imshow(rho)
plt.title('Real')
plt.xlabel('Tiempo')
plt.ylabel('rho')
plt.subplot(122)
plt.imshow(rho2)
plt.title('Estimado')
plt.xlabel('Tiempo')
plt.ylabel('rho')
plt.tight_layout()
plt.savefig('Estimacion potencia')
plt.show()
def espectral_eff(h_predict_esc,rho2):
    R=np.zeros([1,h_predict_esc.shape[1]])
    for j in range(h_predict_esc.shape[1]):
        for i in range(h_predict_esc.shape[0]):
            R[0,j]=R[0,j]+np.log2(1+rho2[i,j]*h_predict_esc[i,j])
    return np.squeeze(R/h_predict_esc.shape[0])
R_real=espectral_eff(h,rho)
plt.plot(R_real,label="Real")
R_est=espectral_eff(hi,rho2)
plt.plot(R_est,label="Estimado")
plt.legend()
plt.title('Eficiencia Espectral')
plt.xlabel('Tiempo')
plt.ylabel('R(t)')
plt.savefig('R.jpg')
plt.show()
error = np.abs(R_real-R_est)/R_real *100
plt.plot(error)
plt.ylabel('Porcentaje error')
plt.xlabel('Tiempo')
plt.title('Error porcentual relativo')
plt.savefig('error porcentual relativo')
plt.show()
w = pd.read_csv('/kaggle/input/wwiener/w.csv', header=None)
w_vector = np.squeeze(w.values)
w_vector.shape
w_flip = np.flip(w_vector) # cambiar el orden de los indices
inputs_wiener = np.squeeze(inputs_test)
print(inputs_wiener.shape,outputs_test.shape)

h_predict = np.zeros([512,924])
for i in range(100,1024):
    data_input = h[:,i-100:i]
    data_input = (11.769/2)*(data_input+1)
    h_predict[0:512,i-100] = np.dot(data_input,w_flip)
plt.imshow(h_predict)
plt.colorbar()
r2_score(outputs_test,predict_wienner)
