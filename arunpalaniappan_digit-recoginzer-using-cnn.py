import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt, matplotlib.image as mpimg 
import seaborn as sns
print ('Done')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print ('Done')
digits = train['label'].values
pixels = train.drop(labels = ['label'],axis = 1).values
del train
sns.countplot(digits)
pixels = pixels.reshape(-1,28,28)
pixels.shape
plt.imshow(pixels[0])
plt.title('Before bit changing '+str(digits[0]))
plt.show()
pixels[pixels>0] = 1
plt.imshow(pixels[0])
plt.title('After bit changing '+str(digits[0]))
plt.show()
plt.figure(figsize = (8,8))

plt.subplot(2,2,1)
plt.imshow(pixels[0])

plt.subplot(2,2,2)
plt.imshow(pixels[1])

plt.subplot(2,2,3)
plt.imshow(pixels[2])

plt.subplot(2,2,4)
plt.imshow(pixels[3])
from sklearn.model_selection import train_test_split
train_images,test_images,train_labels, test_labels = train_test_split(pixels,digits, test_size = 0.2, random_state = 0)
import tensorflow as tf
from tensorflow import keras
print ('Done')
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape = (28,28)))
model.add(keras.layers.Dense(128,activation = tf.nn.relu))
model.add(keras.layers.Dense(10,activation = tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
prediction = model.predict_classes(test_images)