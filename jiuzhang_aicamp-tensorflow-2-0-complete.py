import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
print("Your current tensorflow version is: ", tf.__version__)
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
x = tf.Variable(3, name='x')# TODO
y = tf.Variable(4, name='y')# TODO
f = x * x * y + y + 2# TODO
# f
#TODO change the input value of x=0
f = x * x * y + y + 2
# f
# Version 1's code
# sess = tf.Session()# TODO
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)#TODO
# print(result)
# sess.run(x)
constant_tensor = tf.constant([[1, 2], [3, 4]])
# constant_tensor
# constant_tensor.shape
# constant_tensor.numpy()
# np.square(constant_tensor)
# result = constant_tensor + 6
# tf.sqrt(result) ??     ## result = tf.cast(result, tf.float32)

# tf.matmul(result, result)
# np.dot(result, result)
string_example = tf.constant("six_little_phoenix")
# string_example
# tf.strings.length(string_example)

IMAGE_CLASSES = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

import matplotlib.pyplot as plt
np.random.seed(1)
tf.random.set_seed(1)
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
x_train, x_test = X_train / 255.0, X_test / 255.0
img_size = 28
for img, label in zip(x_train[:10], y_train[:10]):
    plt.imshow(img.reshape(img_size,img_size),cmap='gray')
    plt.title(IMAGE_CLASSES[label])
    plt.show()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
# TODO
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

model_json = model.to_json()
with open("fashion_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("fashion_model.h5")
