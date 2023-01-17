import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
mnist = keras.datasets.mnist
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
import matplotlib.pyplot as plt
train_data = pd.read_csv('../input/train.csv')
train_images = np.array(train_data.drop('label', axis=1))
train_labels = np.array(train_data['label']).reshape(train_images.shape[0], 1)

#reshape images from (784,) to (28,28)
train_images = np.array(train_images.reshape((-1,28, 28,1)) )

#spliting the data
train_images,test_images ,train_labels, test_labels= train_test_split(train_images, train_labels)

print('Trainning data',train_images.shape)
print('train labels',train_labels.shape)

print('Test data',test_images.shape)
print('test labels',test_labels.shape)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(train_labels)
train_labels_hot = np.array(enc.transform(train_labels).toarray())
enc2 = OneHotEncoder()
enc2.fit(test_labels)
train_labels_hot = np.array(enc2.transform(test_labels).toarray())
train_images = train_images/255.
test_images = test_images/255.
plt.imshow(train_images[5].reshape(28, 28))
#from keras.layers import Conv2D
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32,(3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128,(3,3), activation=tf.nn.relu),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 
epochs=15
history=model.fit(train_images, train_labels,batch_size=128, epochs=epochs,validation_data=(test_images,test_labels)) 
history_dict=history.history
history_dict.keys()
epochs=range(1,len(history_dict['acc'])+1)
acc_values=history_dict['acc']
loss_values=history_dict['loss']
valid_acc=history_dict['val_acc']
valid_loss=history_dict['val_loss']
plt.clf()
plt.plot(epochs,loss_values,'b+-',label='T_loss')
plt.plot(epochs,valid_loss,'r+-',label='V_loss')
plt.legend()
plt.show()
plt.clf()
plt.plot(epochs,acc_values,'b.-',label='T_acc')
plt.plot(epochs,valid_acc,'r.-',label='V_acc')
plt.legend()
plt.show()
#model.summary()
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Validation accuracy: ', test_acc)
print('    Validation loss: ', test_loss)
testing_images = pd.read_csv('../input/test.csv')
print("Testing images: ",testing_images.shape)

testing_images = np.array(testing_images)
testing_images = np.array(testing_images.reshape((-1,28, 28,1)) )

labels_predicted=model.predict(testing_images)
print("Testing labels predicted: ",labels_predicted.shape)

sample_sub = pd.read_csv('../input/sample_submission.csv')
print("Sample submission: ",sample_sub.shape)
labels_predicted = np.argmax(labels_predicted, axis=1)
my_submission = pd.DataFrame({'ImageId': np.array(range(0,labels_predicted.shape[0])), 'Label': labels_predicted})
print(my_submission.shape)
print(my_submission.head())
