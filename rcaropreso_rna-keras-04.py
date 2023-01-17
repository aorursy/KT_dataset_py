#CÉLULA KE-LIB-01
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import seaborn as sns
import os
from matplotlib import pyplot as plt
%matplotlib inline
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02

#CÉLULA KE-LIB-03

#CÉLULA KE-LIB-04
np.random.seed(4)
tf.set_random_seed(13)
#CÉLULA KE-LIB-05

#CÉLULA KE-LIB-06
batch_size = 128
max_epochs = 50
print("Iniciando treinamento... ")

#CÉLULA KE-LIB-07

#CÉLULA KE-LIB-8
# Salvando modelo em arquivo
print("Salvando modelo em arquivo \n")
mp = ".\\mnist_model.h5"
model.save(mp)
print("Usando o modelo para previsão de dígitos para a imagem: ")
unknown = np.zeros(shape=(28,28), dtype=np.float32)
for row in range(5,23): unknown[row][9] = 180 # vertical line
for rc in range(9,19): unknown[rc][rc] = 250 # diagonal line
plt.imshow(unknown, cmap=plt.get_cmap('gray_r'))
plt.show()

unknown = unknown.reshape(1, 28,28,1)
predicted = model.predict(unknown)
print("\nO valor do dígito previsto é: ")
print(predicted)
# 7. Pos-processamento
str = ['zero', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']

index = np.argmax(predicted[0])
digit = str[index]
print(digit)