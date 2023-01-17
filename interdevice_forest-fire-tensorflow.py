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
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
df_amazon = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
df_amazon.head()
dfa = df_amazon.groupby(['state', 'year'], as_index=False).agg({'number': 'sum'})
dfa.head()
base = dfa.iloc[:, 2].values
plt.plot(base)
periodos = 27

previsao_futura = 1 # horizonte das previsões
X = base[0:(len(base) - (len(base) % periodos))]
X_batches = X.reshape(-1, periodos, 1)
y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]
y_batches = y.reshape(-1, periodos, 1)
X_teste = base[-(periodos + previsao_futura):]

X_teste = X_teste[:periodos]

X_teste = X_teste.reshape(-1, periodos, 1)
y_teste = base[-(periodos):]

y_teste = y_teste.reshape(-1, periodos, 1)
import tensorflow as tf
tf.reset_default_graph()
entradas = 1

neuronios_oculta = 100

neuronios_saida = 1
xph = tf.placeholder(tf.float32, [None, periodos, entradas])

yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])
celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)

celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)
saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)

erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)

otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    

    for epoca in range(1000):

        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})

        if epoca % 100 == 0:

            print(epoca + 1, ' erro: ', custo)

            

    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})
import numpy as np
y_teste.shape

y_teste2 = np.ravel(y_teste)
previsoes2 = np.ravel(previsoes)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_teste2, previsoes2)

mae
plt.plot(y_teste2, markersize = 10, label = 'Valor real')

plt.plot(previsoes2, label = 'Previsões')

plt.legend()