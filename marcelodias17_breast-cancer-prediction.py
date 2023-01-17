# Importando as funções necessárias

from __future__ import print_function



import tensorflow as tf 

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import RMSprop



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# Lendo o dataset e imprimindo uma amostra

breast_cancer = pd.read_csv('../input/data.csv')

breast_cancer.head()
# Descrevendo as informações dos tipos de dados do dataset

breast_cancer.info()
# Verificando o tamanho do dataset (569 linhas / 33 colunas)

breast_cancer.shape
# Gerando estatísticas descritivas do dataset que apresentam um resumo dos dados, excluindo valores nulos

breast_cancer.describe()
# Contagem de registros agrupados pela coluna 'diagnosis', que representa o diagnóstico (B = Benigno / M = Maligno)

breast_cancer.groupby('diagnosis').size()
# Contagem de registros nulos por coluna

breast_cancer.isnull().sum()
# As colunas 'id' e 'Unnamed: 32' não são úteis para a análise e serão descartadas 

feature_names = breast_cancer.columns[2:-1]

x = breast_cancer[feature_names]

# A coluna 'diagnosis' é a característica que vamos prever

y = breast_cancer.diagnosis
# Transforma os dados da coluna 'diagnosis' para valores binários (M = 1 / B = 0)

class_le = LabelEncoder()

y = class_le.fit_transform(breast_cancer.diagnosis.values)
# Gera uma matriz de correlação (heatmap) que fornece informações úteis sobre a relação entre cada variável do conjunto de dados

sns.heatmap(

    data=x.corr(),

    annot=True,

    fmt='.2f',

    cmap='RdYlGn'

)



fig = plt.gcf()

fig.set_size_inches(20, 16)



plt.show()
# Obtendo os conjuntos de treino e teste, separando 32% do conjunto de dados para teste (test_size=0.32) e o restante para treino

x_train, x_test, y_train, y_test = train_test_split(

    x,

    y,

    random_state=42,

    test_size=0.32

)



print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
# Implementação da rede neural, utilizando 2 classes (diagnosis) e 200 épocas

batch_size = 64

num_classes = 2

epochs = 200



# Transformando os dados de entrada para float32

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# Convertendo os vetores das classes em matrizes de classificação binárias

y_train = tf.keras.utils.to_categorical(y_train, num_classes)

y_test = tf.keras.utils.to_categorical(y_test, num_classes)



# Definição da arquitetura do modelo

model = Sequential()

# Camadas do modelo

model.add(tf.keras.layers.Dense(100, input_dim=30, activation='sigmoid'))

model.add(tf.keras.layers.Dense(25, input_dim=30, activation='relu'))

model.add(tf.keras.layers.Dense(2, activation='softmax'))



# Fim - Definição da arquitetura do modelo



model.summary()



model.compile(loss='categorical_crossentropy',

              optimizer=RMSprop(0.0001),

              metrics=['accuracy'])



# Treinamento do modelo 

H = model.fit(x_train, y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1,

                    validation_data=(x_test, y_test))



# Avaliação do modelo no conjunto de teste

score = model.evaluate(x_test, y_test, verbose=1)



print('Test loss:', score[0])

print('Test accuracy:', score[1])



# Plotando 'loss' e 'accuracy' para os datasets 'train' e 'test'

plt.figure()

plt.plot(np.arange(0,epochs), H.history["loss"], label="train_loss")

plt.plot(np.arange(0,epochs), H.history["val_loss"], label="val_loss")

plt.plot(np.arange(0,epochs), H.history["acc"], label="train_acc")

plt.plot(np.arange(0,epochs), H.history["val_acc"], label="val_acc")

plt.title("Acurácia")

plt.xlabel("Épocas #")

plt.ylabel("Loss/Accuracy")

plt.legend()

plt.show()