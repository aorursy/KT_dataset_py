# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Arquivo para treinamento do modelo

df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

print(df_train.shape)

df_train.head()
# Arquivo para teste do modelo

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df_test.head()
# 5 ultimas linhas

df_test.tail()
# iloc[<linhas>, <colunas>]

# todos os dados de linha e coluna, com exceção da primeira coluna

df_train.iloc[:,1:].values
# armazena todas as colunas referentes à pixel na variavel X_train

X_train = (df_train.iloc[:,1:].values).astype('float32')

# armazena a coluna label - target, na variavel X_target

X_target = df_train['label'] #df_train.iloc[:,0].values.astype('int32')



# armazena todas as colunas de teste na variavel X_test

X_test = df_test.values.astype('float32')
# converte o dataframe X_train (num_images, img_rows, img_cols)

# 28x28 pixels

X_train = X_train.reshape(X_train.shape[0], 28, 28)



import matplotlib.pyplot as plt

%matplotlib inline



for i in range(1, 3):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(X_target[i]);
for i in range(3, 6):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(X_target[i]);
for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(X_target[i]);
# apresenta estatísticas descritivas do dataframe

df_train.describe()
# Standardize the data

from sklearn.preprocessing import StandardScaler

X = df_train.values

X_std = StandardScaler().fit_transform(X)



# Calculating Eigenvectors and eigenvalues of Cov matirx

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the eigenvalue, eigenvector pair from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# plot de alguns numeros

plt.figure(figsize=(14,12))

for digit_num in range(0,70):

    plt.subplot(7,10,digit_num+1)

    grid_data = train.iloc[digit_num].values.reshape(28,28)  # reshape from 1d to 2d pixel array

    plt.imshow(grid_data, interpolation = "none", cmap = "afmhot")

    plt.xticks([])

    plt.yticks([])

plt.tight_layout()
from sklearn.cluster import KMeans

# define o numero de clusters do KMeans (10 de classe)

kmeans = KMeans(n_clusters=10)

# computa os centros de cluster e prever o índice de cluster para cada amostra

X_clustered = kmeans.fit_predict(X_5d)



# plota pontos por classe 

trace_Kmeans = go.Scatter(x=X_5d[:, 0], y= X_5d[:, 1], mode="markers",

                    showlegend=False,

                    marker=dict(

                            size=8,

                            color = X_clustered,

                            colorscale = 'Portland',

                            showscale=False, 

                            line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        )

))



layout = go.Layout(

    title= 'KMeans Clustering',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)



data = [trace_Kmeans]

fig1 = dict(data=data, layout= layout)

# 

py.iplot(fig1, filename="svm")
from keras.models import Sequential

from keras.utils import np_utils

from keras.layers.core import Dense, Activation, Dropout



# reseta valor das variaveis X_train e X_target pois foram utilizadas a cima

X_train = (df_train.iloc[:,1:].values).astype('float32')

X_target = df_train['label']

X_test = df_test.astype('float32')



# converte o vetor de classe (labels) em uma matriz de classe binária

y_train = np_utils.to_categorical(X_target)



# pré-processamento

# divide valor dos dataframes X pela média máxima

scale = np.max(X_train)

X_train /= scale

X_test /= scale



# subtrai valor dos dataframes X pela média

mean = np.std(X_train)

X_train -= mean

X_test -= mean



input_dim = X_train.shape[1]

nb_classes = y_train.shape[1]



# cria um modelo Sequential

model = Sequential()

# define o formato de entrada

model.add(Dense(128, input_dim=input_dim))

# função de ativação de Unidade linear retificada (Rectified Linear Unit)

model.add(Activation('relu'))

# aplica o Dropout ao input - eliminamos algumas unidades aleatoriamente (15% de chance de eliminação)

model.add(Dropout(0.15))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Dense(nb_classes))

# função de ativação Softmax

model.add(Activation('softmax'))



# configura o modelo para treinamento

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



print("Treinamento do modelo")

# epochs - cutoff, separa o treino em fases distintas

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1, verbose=2)



print("Gera predições")

_predict = model.predict_classes(X_test, verbose=0)
# cria dataframe para geração do arquivo de saida

df_predict = pd.DataFrame()

df_predict["ImageId"] = np.arange(len(_predict))+1

df_predict["Label"] = _predict



df_predict.head()
# Gerar arquivo csv de saida

df_predict.to_csv('gender_submission.csv', index=False)