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

#CÉLULA KE-LIB-04

#CÉLULA KE-LIB-05
#Montando a rede neural

#CÉLULA KE-LIB-06
#Treinamento

#CÉLULA KE-LIB-07
#Treinamento

#CÉLULA KE-LIB-08
# 5 Avaliação do modelo

#CÉLULA KE-LIB-9
# Salvando modelo em arquivo
print("Salvando modelo em arquivo \n")
mp = ".\\boston_model.h5"
model.save(mp)
# 7. Usando modelo (operação)
np.set_printoptions(precision=4)
unknown = np.full(shape=(1,13), fill_value=0.6, dtype=np.float32)
unknown[0][3] = -1.0 # encodando o booleano
predicted = model.predict(unknown)
print("Usando o modelo para previsão de preço médio de casa para as caracteristicas: ")
print(unknown)
print("\nO preço médio será [dolares]: ")
print(predicted * 10000)