import tensorflow as tf
import numpy
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train.shape

x_train = x_train.reshape(60000,28,28,1)
x_train.shape

x_test = x_test.reshape(10000,28,28,1)
x_test.shape
x_train = x_train/255.0
x_test = x_test/255.0
model = tf.keras.Sequential([
    
    
tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
    
 tf.keras.layers.Dense(128,activation='relu'),
tf.keras.layers.Dense(10,activation='softmax')
    
    
    
    
    
    
    
    
    
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs=20)
loss = model.evaluate(x_test,y_test)
