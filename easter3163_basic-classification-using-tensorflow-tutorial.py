import pandas as pd
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Y = train['label']
X = train.drop(labels=["label"], axis=1)
X = X / 255.0
test = test / 255.0
from sklearn.model_selection import train_test_split
random_seed=0
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow import keras
plt.figure()
Xshow = X_train.values.reshape(-1,28,28,1)
plt.imshow(Xshow[0][:,:,0])
plt.colorbar()
plt.grid(False)
X_train.shape
plt.figure(figsize=(10,10))
class_names = ['0','1','2','3','4','5','6','7','8','9']
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(Xshow[i][:,:,0], cmap=plt.cm.binary)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
X_val = X_val.values.reshape(-1,28,28)
X_train = X_train.values.reshape(-1,28,28)
test = test.values.reshape(-1,28,28)
model.fit(X_train,Y_train, epochs=5)
test_loss, test_acc = model.evaluate(X_val, Y_val)
print("Test Accuracy: {}".format(test_acc))
X = X.values.reshape(-1,28,28)
model.fit(X,Y, epochs=5)
predictions = model.predict(test)
predictions = np.argmax(predictions, axis=1)
submission = pd.read_csv('../input/sample_submission.csv')
submission["Label"] = predictions
submission.head()
submission.to_csv('./simpleMNIST.csv', index=False)