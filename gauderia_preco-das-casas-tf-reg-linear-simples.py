import os

print(os.listdir("../input"))
import pandas as pd

df = pd.read_csv('../input/kc_house_data.csv')
df.head()
df.count()
df.shape
x = df.iloc[:,5] #todas as linhas da coluna 5

x= df.iloc[:,5].values #transformando em np array

x =x.reshape(-1, 1) #-1 sign q n vai mexer na coluna, vai apenas add outra
x
y = df.iloc[:,2:3].values # p n precisar fazer reshape

y.shape
from sklearn.preprocessing import StandardScaler #escalonando a gurizada

scaler_x = StandardScaler()

scaler_y = StandardScaler()

x = scaler_x.fit_transform(x)

y = scaler_y.fit_transform(y)
import matplotlib.pyplot as plt

%matplotlib inline

plt.scatter(x,y)
import numpy as np

np.random.seed(1)

np.random.rand(2)
import tensorflow as tf
b0 = tf.Variable(0.41)

b1 = tf.Variable(0.72)
batch_size = 32 #pega de 32 em 32 até chegar no total de 21613

xph = tf.placeholder(tf.float32, [batch_size, 1])

yph = tf.placeholder(tf.float32, [batch_size, 1])
y_model = b0 + b1 * xph

erro = tf.losses.mean_squared_error(yph, y_model)

otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro)

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)

    for i in range (10000):

        indices = np.random.randint(len(x), size=batch_size)

        feed = {xph : x[indices], yph : y[indices]}

        sess.run(treinamento, feed_dict=feed)

    b0_final, b1_final = sess.run([b0,b1])
b0_final
b1_final
previsoes = b0_final + b1_final * x
previsoes
plt.plot(x,y, 'o')

plt.plot(x, previsoes, color='red')
y1 = scaler_y.inverse_transform(y)

previsoes1 = scaler_y.inverse_transform(previsoes)
y1
previsoes1
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y1, previsoes1)

mae #pode errar isso p cima ou p baixo
teste = scaler_y.inverse_transform(b0_final + b1_final * scaler_x.transform([[800]]))

teste