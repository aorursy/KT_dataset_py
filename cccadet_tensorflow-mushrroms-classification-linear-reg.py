#Auxílio do Tutorial: https://matheusfacure.github.io/2017/05/12/tensorflow-essencial/



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import tensorflow as tf

import gzip

import pickle

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import os # para criar pastas

from sklearn.metrics import r2_score, accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/mushrooms.csv')

df.head()
le = preprocessing.LabelEncoder()

df_encoded = df.apply(le.fit_transform)

list(le.classes_)

#list(le.inverse_transform([2, 2, 1]))



df_encoded.astype(float)

scaler = MinMaxScaler()

df_encoded[df_encoded.columns] = scaler.fit_transform(df_encoded[df_encoded.columns])

df_encoded.head()



X = df_encoded.drop(['class'], axis=1)

y = df_encoded['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Formato dos dados:', X_train.shape, y_train.shape)

# definindo constantes

lr = 1e-2 # taxa de aprendizado

n_iter = 2501 # número de iterações de treino

n_inputs = X_train.shape[1] # número de variáveis independentes

n_outputs = 1 # número de variáveis dependentes



graph = tf.Graph() # isso cria um grafo

with graph.as_default(): # isso abre o grafo para que possamos colocar operações e variáveis dentro dele.

    tf.set_random_seed(1)

    

    # adiciona as variáveis ao grafo

    W = tf.Variable(tf.truncated_normal([n_inputs, n_outputs], stddev=.1), name='Weight')

    b = tf.Variable(tf.zeros([n_outputs]), name='bias')





    ######################################

    # Monta o modelo de regressão linear #

    ######################################



    # Camadas de Inputs

    x_input = tf.placeholder(tf.float32, [None, n_inputs], name='X_input')

    y_input = tf.placeholder(tf.float32, [None, n_outputs], name='y_input')



    # Camada Linear

    y_pred = tf.add(tf.matmul(x_input, W), b, name='y_pred')



    # Camada de custo ou função objetivo

    EQM = tf.reduce_mean(tf.square(y_pred - y_input), name="EQM")



    # otimizador

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(EQM)



    # inicializador

    init = tf.global_variables_initializer()



    # para salvar o modelo treinado

    saver = tf.train.Saver()

# criamos uma pasta para salvar o modelo

if not os.path.exists('tmp'):

    os.makedirs('tmp')



# abrimos a sessão tf

with tf.Session(graph=graph) as sess:

    sess.run(init) # iniciamos as variáveis

    

    # cria um feed_dict

    feed_dict = {x_input: X_train, y_input: y_train.values.reshape(-1,1)}

    

    # realizamos as iterações de treino

    for step in range(n_iter + 1):

        

        # executa algumas operações do grafo

        _, l = sess.run([optimizer, EQM], feed_dict=feed_dict)

        

        if (step % 500) == 0:

            print('Custo na iteração %d: %.2f \r' % (step, l), end='')

            saver.save(sess, "./tmp/my_model.ckpt")

# novamente, abrimos a sessão tf

with tf.Session(graph=graph) as sess:

    

    # restauramos o valor das variáveis 

    saver.restore(sess, "./tmp/my_model.ckpt", )

    

    # rodamos o nó de previsão no grafo

    y_hat = sess.run(y_pred, feed_dict={x_input: X_test})

    

    print('\nR2: %.3f' % r2_score(y_pred=y_hat, y_true=y_test))

    print('\nAccuracy %.3f' % accuracy_score(y_test, y_hat.round(), normalize=True))
