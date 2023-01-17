import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
def plot(h):
    plt.style.use('seaborn-darkgrid')

    epochs = h.epoch

    acc = h.history['accuracy']
    val_acc = h.history['val_accuracy']

    loss = h.history['loss']
    val_loss = h.history['val_loss']

    fig = plt.figure(figsize = (6,4),dpi = 100)
    ax = fig.add_axes([1,1,1,1])

    ax.plot(epochs,acc,label = 'Accuracy')
    ax.plot(epochs,val_acc,label = 'Val Accuracy')

    ax.plot(epochs,loss,label = 'Loss')
    ax.plot(epochs,val_loss,label = 'Val Loss')
    
    ax.set(xlabel = 'Number of Epochs',ylabel = 'Measure',title = 'Learning Curve')

    ax.legend()
#Create Model
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),                                    
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(10, activation='softmax')
])

#Define Optimizer
sgd = tf.keras.optimizers.SGD(lr = 0.001)

#Compile Model
model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])
tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False,dpi=100)
#Train Model
h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3), activation='relu',kernel_initializer = 'he_normal', input_shape=(28, 28, 1)),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu',kernel_initializer = 'he_normal'),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), activation='relu',kernel_initializer = 'he_normal'),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu',kernel_initializer = 'he_normal'),                                    
                                    tf.keras.layers.Dense(128, activation='relu',kernel_initializer = 'he_normal'),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.PReLU(),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.PReLU(),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.PReLU(),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.PReLU(),

                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.PReLU(),
                                    
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.ELU(),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.ELU(),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.ELU(),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.ELU(),

                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal'),
                                    tf.keras.layers.ELU(),
                                    
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.BatchNormalization(),                                    
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001,clipvalue = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001,momentum = 0.90,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.001,momentum = 0.90,nesterov = True,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


rms = tf.keras.optimizers.RMSprop(lr = 0.001,rho = 0.90,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = rms,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


adam = tf.keras.optimizers.Adam(lr = 0.001,beta_1 = 0.9,beta_2=0.999,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


admx = tf.keras.optimizers.Adamax(lr = 0.001,beta_1 = 0.9,beta_2=0.999,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = admx,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


ndm = tf.keras.optimizers.Nadam(lr = 0.001,beta_1 = 0.9,beta_2=0.999,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = ndm,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0,decay = 1e-4)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
def exp_decay_function(epoch):
    return 0.01*(0.1**(epoch / 20))

lr = tf.keras.callbacks.LearningRateScheduler(exp_decay_function)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32,callbacks = [lr])
plot(h)
def piecewise_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001

lr = tf.keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32,callbacks = [lr])
plot(h)
lr = tf.keras.callbacks.ReduceLROnPlateau(factor = 0.5, patience = 5)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32,callbacks = [lr])
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',kernel_regularizer = tf.keras.regularizers.l2(0.001),use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',kernel_regularizer = tf.keras.regularizers.l2(0.001),use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.Dropout(0.3),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',use_bias = False),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(32, (3,3),kernel_initializer = 'he_normal',input_shape=(28, 28, 1)),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3),kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2, 2),

                                    tf.keras.layers.Conv2D(64, (3,3), kernel_initializer = 'he_normal'),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    tf.keras.layers.MaxPooling2D(2,2),
  
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(256, kernel_initializer = 'he_normal',kernel_constraint = tf.keras.constraints.max_norm(1.),use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),

                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(128, kernel_initializer = 'he_normal',kernel_constraint = tf.keras.constraints.max_norm(1.),use_bias = False),
                                    tf.keras.layers.LeakyReLU(alpha = 0.2),
                                    
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(10, activation='softmax')
])


sgd = tf.keras.optimizers.SGD(lr = 0.01,clipnorm = 1.0)

model.compile(loss = 'sparse_categorical_crossentropy',
              optimizer = sgd,
              metrics = ['accuracy'])

h = model.fit(training_images,training_labels,
          epochs=25,batch_size = 32,steps_per_epoch = 60000//32,
          validation_data = (test_images,test_labels),validation_steps = 10000//32)
plot(h)