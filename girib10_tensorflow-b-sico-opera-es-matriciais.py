import tensorflow as tf
tf.InteractiveSession()
a = tf.zeros(2)
a.eval()
a = tf.zeros((2, 3))
a.eval()
b = tf.ones((2,2,2))
b.eval()
b = tf.fill((2, 2), value=5.) #preenche uma matriz 2x2 com 5s
b.eval()
a = tf.constant(3) #cria uma constante
a.eval()
a = tf.random_normal((2, 2), mean=0, stddev=1) #preenche com números aleatórios de média 0 e desvio 1
a.eval()
a = tf.random_uniform((2, 2), minval=-2, maxval=2)
a.eval()
c = tf.fill((2,2), 2.)
d = tf.fill((2,2), 7.)
e = c * d #multiplicação elemento a elemento 
e.eval()