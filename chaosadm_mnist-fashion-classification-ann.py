import pandas as pd 
import numpy as np 
import tensorflow as tf 

%load_ext tensorboard
(X_train, y_train) , (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data() 
print(X_train[0])
print(y_train[0:5])
# | Label | Description |
# |-------|-------------|
# | 0     | T-shirt/top |
# | 1     | Trouser     |
# | 2     | Pullover    |
# | 3     | Dress       |
# | 4     | Coat        |
# | 5     | Sandal      |
# | 6     | Shirt       |
# | 7     | Sneaker     |
# | 8     | Bag         |
# | 9     | Ankle boot  |
# Preprocessing 

X_train = X_train/255  # Bring down value of each pixel b/w 0 and 1

ann = tf.keras.models.Sequential([
    tf.keras.layers.Reshape(target_shape=(28*28,), input_shape=(28,28)), # Input 
    tf.keras.layers.Dense(units = 128, activation='relu'), # Hidden 
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units = 128, activation='relu'), # Hidden
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units = 64, activation='relu'), # Hidden
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units = 10, activation='softmax') # Output
])
ann.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
X_train = tf.keras.utils.normalize(X_train)
y_train = tf.one_hot(y_train, depth= 10)  # One Hot Encoding 
y_test = tf.one_hot(y_test, depth= 10)
from datetime import datetime 

# logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

ann.fit(
    X_train, y_train, 
    batch_size=128, 
    epochs=30,
    validation_data=(X_test, y_test)
)
X_test = X_test/255 
X_test = tf.keras.utils.normalize(X_test)
predictions = ann.predict(X_test)
predictions[0]
import matplotlib.pyplot as plt 

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

rand = np.random.randint(0, len(predictions)+1)
plt.imshow(X_test[rand]) 
print("Predicted: ",class_names[predictions[rand].argmax()]) 
print("Real: ", class_names[np.array(y_test[rand]).argmax()])
ann.summary()

