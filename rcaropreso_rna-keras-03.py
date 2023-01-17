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
np.random.seed(4)
tf.set_random_seed(13)
#CÉLULA KE-LIB-03
dfCleveland = pd.read_csv('cleveland_train.csv', header=None)
dfCleveland


X_train = dfCleveland[np.arange(0,18)]
y_train = dfCleveland[18]

#CÉLULA KE-LIB-04
#Montando a rede neural

#CÉLULA KE-LIB-05
#Treinamento
batch_size = 8
max_epochs = 2000
print("Iniciando treinamento... ")


#CÉLULA KE-LIB-06

#CÉLULA KE-LIB-7
# Salvando modelo em arquivo
print("Salvando modelo em arquivo \n")
mp = ".\\cleveland_model.h5"
model.save(mp)
# 7. Usando modelo (operação)
np.set_printoptions(precision=4)
unknown = np.array([[0.75, 1, 0, 1, 0, 0.49, 0.27, 1, -1, -1, 0.62, -1, 0.40, 0, 1, 0.23, 1, 0]], dtype=np.float32)
predicted = model.predict(unknown)
print("Usando o modelo para previsão de doença cardíaca para as caracteristicas: ")
print(unknown)
print("\nO valor de previsão diagnóstico é (0=sem doença, 1=com doença): ")
print(predicted)
# 8. Pos-processamento
