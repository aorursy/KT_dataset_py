import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#Variables de Train
num_epocas =50
batch_size = 100
learning_rate = 1e-3
n_datos = mnist.train.num_examples
num_batch = int(n_datos / batch_size)
#Variables del dataset
tam_imagen = 784 # Imagenes de 28x28 pixels
tam_latente = 2
tam_hidden_layer = 700
x = tf.placeholder(tf.float32, shape=[None, tam_imagen]) #Entrada de datos, imagenes
#Semilla aleatoria
tf.set_random_seed(0)
#Pesos y biases
#Primera capa oculta
W_encoder1 = tf.Variable(tf.random_normal([tam_imagen, tam_hidden_layer], stddev= tf.pow(float(tam_imagen), -0.5)))
b_encoder1 = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

#Segunda capa: salida encoder
W_encoder_out = tf.Variable(tf.random_normal([tam_hidden_layer, tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))
b_encoder_out = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

W_z_var = tf.Variable(tf.random_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_var = tf.Variable(tf.random_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

W_z_mean = tf.Variable(tf.random_normal([tam_hidden_layer,tam_latente], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_z_mean = tf.Variable(tf.random_normal([tam_latente], stddev=tf.pow(float(tam_latente), -0.5)))

#Model del Encoder
encoder_capa1 = tf.matmul(x, W_encoder1) + b_encoder1
encoder_capa1 = tf.nn.relu(encoder_capa1)
encoder_out = tf.matmul(encoder_capa1, W_encoder_out) + b_encoder_out
encoder_out = tf.nn.relu(encoder_out)
#Mean
z_mean = tf.matmul(encoder_out,W_z_mean)+b_z_mean
#Std
z_var = tf.matmul(encoder_out, W_z_var)+b_z_var

epsilon = tf.random_normal(tf.shape(z_mean),mean=0.0,stddev=1.0,dtype=tf.float32)
z = z_mean + (tf.multiply(tf.sqrt(tf.exp(z_var)),epsilon))
#Pesos y biases
#Primera capa oculta
W_decoder1 = tf.Variable(tf.random_normal([tam_latente, tam_hidden_layer], stddev=tf.pow(float(tam_latente), -0.5)))
b_decoder1 = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

#Segunda capa oculta
W_decoder2 = tf.Variable(tf.random_normal([tam_hidden_layer, tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))
b_decoder2 = tf.Variable(tf.random_normal([tam_hidden_layer], stddev= tf.pow(float(tam_hidden_layer), -0.5)))

W_decoder_out = tf.Variable(tf.random_normal([tam_hidden_layer, tam_imagen], stddev=tf.pow(float(tam_hidden_layer), -0.5)))
b_decoder_out = tf.Variable(tf.random_normal([tam_imagen], stddev= tf.pow(float(tam_imagen), -0.5)))

#Model del Decoder
decoder_capa1 = tf.matmul(z, W_decoder1) + b_decoder1
decoder_capa1 = tf.nn.relu(decoder_capa1)
decoder_capa2 = tf.matmul(decoder_capa1, W_decoder2) + b_decoder2
decoder_capa2 = tf.nn.relu(decoder_capa2)
decoder_out = tf.matmul(decoder_capa2, W_decoder_out) + b_decoder_out
decoder_out = tf.nn.sigmoid(decoder_out)
#Para evitar que que el calculo de gradientes de error cuando los logaritmos son log(x) con x~0 se suma un sesgo  1e-9
decoder_out = tf.clip_by_value(decoder_out, 1e-9, 1 - 1e-9)
likelihood = tf.reduce_sum(x * tf.log(decoder_out) + (1 - x) * tf.log(1 - decoder_out), 1)
#Divergencia KL:  -D_KL(q(z)||p(z))
KL = (1/2) * tf.reduce_sum(1 + z_var - tf.square(z_mean) - tf.exp(z_var), 1)

#El likelihood se escala al tam de los batches
ELBO =  tf.reduce_mean(KL + likelihood)
#Como nuestro objetivo es maximizar el ELBO vamos en sentido positivo al gradiente 
function_coste = -ELBO
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-08)
train_step = optimizer.minimize(function_coste)
def generador(epoca,ejemplos_test):
        decoder_capa1 = tf.matmul(z, W_decoder1) + b_decoder1
        decoder_capa1 = tf.nn.relu(decoder_capa1)
        decoder_capa2 = tf.matmul(decoder_capa1, W_decoder2) + b_decoder2
        decoder_capa2 = tf.nn.relu(decoder_capa2)
        decoder_out = tf.matmul(decoder_capa2, W_decoder_out) + b_decoder_out
        decoder_out = tf.nn.sigmoid(decoder_out)        
        datos = session.run(decoder_out, feed_dict={x: ejemplos_test})
        if(epoca ==0):
            plt.subplot((num_epocas//10) +1,10,1)
            plt.title("Input")
            img = plt.imshow(ejemplos_test[0].reshape(28, 28))            
            img.set_cmap('gray')
            plt.axis('off')
            plt.subplot((num_epocas//10) +1,10,2)
            plt.title("    Decoder")
            img = plt.imshow(datos[0].reshape(28, 28))
            img.set_cmap('gray')
            plt.axis('off')
        else:
            plt.subplot((num_epocas//10) +1 ,10,epoca + 2)
            img = plt.imshow(datos[0].reshape(28, 28))
            img.set_cmap('gray')
            plt.axis('off')
print('Entrenamiento')
session = tf.InteractiveSession()
tf.global_variables_initializer().run()
valores_coste = []
ejemplos_test,_ = mnist.test.next_batch(batch_size)
plt.figure(figsize=(16, 8))
generador(0,ejemplos_test)
num_col = 1
for epoca in range(1, num_epocas+1):   
    average_coste = 0
    inicio = time.time()
    for i in range(num_batch):         
        batch, _ = mnist.train.next_batch(batch_size)
        _, coste = session.run([train_step, function_coste], feed_dict = {x: batch} )
        average_coste = (average_coste + coste)
        valores_coste.append(coste)
    fin = time.time()
    tiempo = fin - inicio
    print('Epoca: %d Tiempo: %f  Loss= %f' % (epoca,tiempo,average_coste/num_batch))
    generador(num_col,ejemplos_test)
    num_col += 1


print('\nTest con %d-D latentes, Numero de epocas %d, Tasa de Aprendizaje %f  y Tam Capas Ocultas %d' % (tam_latente, num_epocas,learning_rate,tam_hidden_layer))
print ("\nDECODIFICACION")
print ("\nInput")
plt.figure()
plt.title("Funcion de coste (Maximiza ELBO)")
plt.plot(valores_coste)
plt.show()
n = 10
input_decoder_2d = tf.placeholder(tf.float32, shape=[None, tam_latente])
canvas = np.empty((28*n, 28*n))
for i in range(n):
    for j in range(n):
        decoder_capa1 = tf.matmul(input_decoder_2d, W_decoder1) + b_decoder1
        decoder_capa1 = tf.nn.relu(decoder_capa1)
        decoder_capa2 = tf.matmul(decoder_capa1, W_decoder2) + b_decoder2
        decoder_capa2 = tf.nn.relu(decoder_capa2)
        decoder_out = tf.matmul(decoder_capa2, W_decoder_out) + b_decoder_out
        decoder_out = tf.nn.sigmoid(decoder_out)        
        imagenes_generadas = session.run(decoder_out, feed_dict={input_decoder_2d: np.random.randn(50, tam_latente)})
        canvas[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = imagenes_generadas[np.random.randint(0, 49)].reshape(28, 28)

plt.figure(figsize=(8, 10))
plt.imshow(canvas,cmap="binary", origin="upper")
plt.tight_layout()
input_image, target = mnist.test.next_batch(mnist.test.num_examples)
input_encoder_2d = tf.placeholder(tf.float32, shape=[None, tam_imagen])
encoder_capa1 = tf.matmul(input_encoder_2d, W_encoder1) + b_encoder1
encoder_capa1 = tf.nn.relu(encoder_capa1)
encoder_out = tf.matmul(encoder_capa1, W_encoder_out) + b_encoder_out
encoder_out = tf.nn.relu(encoder_out)
encoder = session.run(encoder_out, feed_dict={input_encoder_2d: input_image})
z_mean =tf.matmul(encoder,W_z_mean)+b_z_mean
mean = session.run(z_mean, feed_dict={input_encoder_2d: input_image})
colores= []
for i in target:
    indice = np.where(i == 1)
    colores.append(indice[0][0])

plt.figure(figsize=(12, 8)) 
plt.scatter(mean[:, 0], mean[:, 1], s = 15, c=colores)
plt.xlabel('1ª Dimension') 
plt.ylabel('2ª Dimension')  
plt.colorbar()
plt.grid()
%matplotlib inline
from mpl_toolkits.mplot3d import axes3d
input_image, target = mnist.test.next_batch(mnist.test.num_examples)
input_encoder_3d = tf.placeholder(tf.float32, shape=[None, tam_imagen])
encoder_capa1 = tf.matmul(input_encoder_3d, W_encoder1) + b_encoder1
encoder_capa1 = tf.nn.relu(encoder_capa1)
encoder_out = tf.matmul(encoder_capa1, W_encoder_out) + b_encoder_out
encoder_out = tf.nn.relu(encoder_out)
encoder = session.run(encoder_out, feed_dict={input_encoder_3d: input_image})
z_mean =tf.matmul(encoder,W_z_mean)+b_z_mean
mean = session.run(z_mean, feed_dict={input_encoder_3d: input_image})
fig = plt.figure(figsize=(12, 8)) 
ax = fig.add_subplot(111, projection='3d')
colores= []
for i in target:
    indice = np.where(i == 1)
    colores.append(indice[0][0])

plt.scatter(mean[:, 0], mean[:, 1], s = 10 ,c=colores)
plt.xlabel('1ª Dimension') 
plt.ylabel('2ª Dimension') 
plt.autoscale(enable=True, axis='both', tight=None)
plt.colorbar()
plt.grid()
n = 20
primera_dim = np.linspace(-3,3, n)
segunda_dim = np.linspace(-3,3, n)
z_test = tf.placeholder(tf.float32, shape=[None, tam_latente])
canvas = np.empty((28*n, 28*n))
for i, primera in enumerate(primera_dim):
    for j, segunda in enumerate(segunda_dim):
        z_mean = np.array([[primera, segunda]]*batch_size)
        decoder_capa1 = tf.matmul(z_test, W_decoder1) + b_decoder1
        decoder_capa1 = tf.nn.relu(decoder_capa1)
        decoder_capa2 = tf.matmul(decoder_capa1, W_decoder2) + b_decoder2
        decoder_capa2 = tf.nn.relu(decoder_capa2)
        decoder_out = tf.matmul(decoder_capa2, W_decoder_out) + b_decoder_out
        decoder_out = tf.nn.sigmoid(decoder_out)             
        imagenes_generadas = session.run(decoder_out, feed_dict={z_test: z_mean})
        canvas[(n-i-1)*28:(n-i)*28, j*28:(j+1)*28] = imagenes_generadas[0].reshape(28, 28)

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(primera_dim, segunda_dim)
plt.imshow(canvas,cmap="binary", origin="upper")
plt.tight_layout()
%matplotlib inline
from ipywidgets import interactive, Layout

z_test = tf.placeholder(tf.float32, shape=[None, tam_latente])
def generador(primera_dimension, segunda_dimension):
    ejemplo = np.array([[primera_dimension, segunda_dimension]]*batch_size) 
    decoder_capa1 = tf.matmul(z_test, W_decoder1) + b_decoder1
    decoder_capa1 = tf.nn.relu(decoder_capa1)
    decoder_capa2 = tf.matmul(decoder_capa1, W_decoder2) + b_decoder2
    decoder_capa2 = tf.nn.relu(decoder_capa2)
    decoder_out = tf.matmul(decoder_capa2, W_decoder_out) + b_decoder_out
    decoder_out = tf.nn.sigmoid(decoder_out)          
    imagenes_generadas = session.run(decoder_out, feed_dict={z_test: ejemplo})
    imagen = imagenes_generadas[0].reshape(28, 28)
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)
    ax.imshow(imagen, cmap="binary")
    plt.axis("off")

plot = interactive(generador, primera_dimension=(-3, 3,0.01), segunda_dimension=(-3, 3, 0.01))
output = plot.children[0]
output.layout.width = '600px'
output.description='Primera dimension'
output.style = {'description_width': 'initial'}
output = plot.children[1]
output.layout.width = '600px'
output.description='Segunda dimension'
output.style = {'description_width': 'initial'}
plot
import matplotlib.image as mpimg
plt.figure(figsize=(16, 8))
plt.subplot(1,5,1)
plt.title("2-D")
img=mpimg.imread('2D_Espacio_Latente.png')
img = plt.imshow(img)            
img.set_cmap('gray')
plt.axis('off')
plt.subplot(1,5,2)
plt.title("5-D")
img=mpimg.imread('5D_Espacio_Latente.png')
img = plt.imshow(img)            
img.set_cmap('gray')
plt.axis('off')
plt.subplot(1,5,3)
plt.title("10-D")
img=mpimg.imread('10D_Espacio_Latente.png')
img = plt.imshow(img)            
img.set_cmap('gray')
plt.axis('off')
plt.subplot(1,5,4)
plt.title("20-D")
img=mpimg.imread('20D_Espacio_Latente.png')
img = plt.imshow(img)            
img.set_cmap('gray')
plt.axis('off')
plt.subplot(1,5,5)
plt.title("50-D")
img=mpimg.imread('50D_Espacio_Latente.png')
img = plt.imshow(img)            
img.set_cmap('gray')
plt.axis('off')
plt.show()
