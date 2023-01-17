#Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#Variables de Train
num_epocas =5
batch_size = 100
learning_rate = 1e-1
n_datos = mnist.train.num_examples
num_batch = int(n_datos / batch_size)

#Variables del dataset
tam_imagen = 784 # Imagenes de 28x28 pixels
tam_latente = 50
tam_hidden_layer = 500

input_encoder = tf.placeholder(tf.float32, shape=[None, tam_imagen]) #Entrada de datos, imagenes

#Semilla aleatoria
tf.set_random_seed(0)

#ENCODER

#Pesos y biases
#Primera capa oculta
W_encoder1 = tf.Variable(tf.random_normal([tam_imagen, tam_hidden_layer], stddev= tf.pow(float(tam_imagen), -0.5)))
b_encoder1 = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

#Segunda capa oculta
W_encoder2 = tf.Variable(tf.random_normal([tam_hidden_layer, tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))
b_encoder2 = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

W_z_var = tf.Variable(tf.random_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_var = tf.Variable(tf.random_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

W_z_mean = tf.Variable(tf.random_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_mean = tf.Variable(tf.random_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

#Model del Encoder
encoder_capa1 = tf.matmul(input_encoder, W_encoder1) + b_encoder1
encoder_capa1 = tf.nn.relu(encoder_capa1)
encoder_capa2 = tf.matmul(encoder_capa1, W_encoder2) + b_encoder2
encoder_capa2 = tf.nn.relu(encoder_capa2)
#Mean
z_mean = tf.matmul(encoder_capa2,W_z_mean)+b_z_mean


W_clasificador1 = tf.Variable(tf.random_normal([tam_latente, 10], stddev=tf.pow(float(tam_latente), -0.5)))
b_clasificador1= tf.Variable(tf.random_normal([10], stddev=tf.pow(float(tam_latente), -0.5)))
y_clasificador1 = tf.nn.softmax(tf.matmul(z_mean, W_clasificador1) + b_clasificador1)

y_clasificador1_labels = tf.placeholder(tf.float32, [None, 10])

cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(y_clasificador1_labels * tf.log(y_clasificador1), reduction_indices=[1]))
train_step1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy1)

sess1 = tf.InteractiveSession()
tf.global_variables_initializer().run()

print ("Entrenando modelo...")
for epoca in range(1, num_epocas+1):
    average_coste = 0
    for _ in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, coste = sess1.run([train_step1, cross_entropy1], feed_dict={input_encoder: batch_xs, y_clasificador1_labels: batch_ys})
        average_coste = (average_coste + coste)
    print('Epoca: %d Cross Entropy= %f' % (epoca,average_coste/num_batch))

#Test
prediccion1 = tf.equal(tf.argmax(y_clasificador1, 1), tf.argmax(y_clasificador1_labels, 1))
score1 = tf.reduce_mean(tf.cast(prediccion1, tf.float32))
print ("Score")
print(sess1.run(score1, feed_dict={input_encoder: mnist.test.images, y_clasificador1_labels: mnist.test.labels}))
input_clasificador2 = tf.placeholder(tf.float32, shape=[None, tam_imagen]) #Entrada de datos, imagenes
W_clasificador2 = tf.Variable(tf.random_normal([tam_imagen, 10], stddev=tf.pow(float(tam_latente), -0.5)))
b_clasificador2 = tf.Variable(tf.random_normal([10], stddev=tf.pow(float(tam_latente), -0.5)))
y_clasificador2 = tf.nn.softmax(tf.matmul(input_clasificador2, W_clasificador2) + b_clasificador2)

y_clasificador2_labels = tf.placeholder(tf.float32, [None, 10])

cross_entropy2 = tf.reduce_mean(-tf.reduce_sum(y_clasificador2_labels * tf.log(y_clasificador2), reduction_indices=[1]))
train_step2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy2)

sess2 = tf.InteractiveSession()
tf.global_variables_initializer().run()

print ("Entrenando modelo...")
for epoca in range(1, num_epocas+1):
    average_coste = 0
    for _ in range(num_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, coste = sess2.run([train_step2, cross_entropy2], feed_dict={input_clasificador2: batch_xs, y_clasificador2_labels: batch_ys})
        average_coste = (average_coste + coste)
    print('Epoca: %d Cross Entropy= %f' % (epoca,average_coste/num_batch))
    
#Test
prediccion2 = tf.equal(tf.argmax(y_clasificador2, 1), tf.argmax(y_clasificador2_labels, 1))
score2 = tf.reduce_mean(tf.cast(prediccion2, tf.float32))
print ("Score")
print(sess2.run(score2, feed_dict={input_clasificador2: mnist.test.images,y_clasificador2_labels: mnist.test.labels}))
%matplotlib inline
from ipywidgets import interactive, Layout

input_image_test = mnist.test.images

imagenes_ordenadas = np.zeros([10, 784])
for i, imagen in enumerate(input_image_test):    
    indice = np.where(mnist.test.labels[i]==1)[0][0]
    if(imagenes_ordenadas[indice].all() == 0):
        imagenes_ordenadas[indice] = imagen
        


def clasificador(numero):
    plt.figure(figsize=(8, 10))
    plt.subplot(131)
    plt.imshow(imagenes_ordenadas[numero].reshape(28, 28), cmap="gray")
    clasificacion1 = sess1.run(tf.argmax(y_clasificador1, 1), feed_dict={input_encoder: [imagenes_ordenadas[numero]]})
    certeza1 = sess1.run(y_clasificador1, feed_dict={input_encoder: [imagenes_ordenadas[numero]]})
    plt.title("Prediccion: %d  Certeza: %f" % (clasificacion1[0], np.amax(certeza1)) )
    plt.xlabel('Train: Z''s AEVB')    
    plt.subplot(133)
    plt.imshow(imagenes_ordenadas[numero].reshape(28, 28), cmap="gray")
    clasificacion2 = sess2.run(tf.argmax(y_clasificador2, 1), feed_dict={input_clasificador2: [imagenes_ordenadas[numero]]})
    certeza2 = sess2.run(y_clasificador2, feed_dict={input_clasificador2: [imagenes_ordenadas[numero]]})
    plt.title("Prediccion: %d  Certeza: %f" % (clasificacion2[0], np.amax(certeza2)) )
    plt.xlabel('Train: Imagenes Originales')    
    
    

plot = interactive(clasificador, numero=(0,9,1))
output = plot.children[0]
output.layout.width = '600px'
output.description='Numero a clasificar'
output.style = {'description_width': 'initial'}
plot
