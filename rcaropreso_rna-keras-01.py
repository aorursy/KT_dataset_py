#CÉLULA KE-LIB-01
import numpy as np
import keras as K
import tensorflow as tf
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#CÉLULA KE-LIB-02
print("\nIris dataset com Keras/TensorFlow ")

#CÉLULA KE-LIB-03
dfIris = pd.read_csv('iris.csv')
#CÉLULA KE-LIB-04

#CÉLULA KE-LIB-05

#CÉLULA KE-LIB-06

#CÉLULA KE-LIB-07
#Montando a rede neural

#CÉLULA KE-LIB-08
#Treinamento

#CÉLULA KE-LIB-09
# Avaliação do modelo

#CÉLULA KE-LIB-10
# Salvando modelo em arquivo

#CÉLULA KE-LIB-11
# 6. Usando modelo (operação)
np.set_printoptions(precision=4)
unknown = np.array([[6.1, 3.1, 5.1, 1.1]], dtype=np.float32)
predicted = model.predict(unknown)
print("usando o modelo para previsão de espécie para as caracteristicas: ")
print(unknown)
print("\nA espécie é do tipo: ")
print(predicted)


labels = ["setosa", "versicolor", "virginica"]
idx = np.argmax(predicted)
species = labels[idx]
print(species)