# Importing libraries



# Machine Learning

import tensorflow as tf



# Plotting

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



# Data Preparation

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split
train = pd.read_csv(r'../input/digit-recognizer/train.csv')

kaggle_test = pd.read_csv(r'../input/digit-recognizer/test.csv')
# Split data into X and y

X = train[[col for col in train.columns if 'pixel' in col]]

y = train['label']



# Split data into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Preview the data



plt.figure(figsize = (4,4))

gs1 = gridspec.GridSpec(4, 4)

gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 



for i in range(16):

    a = X_train.sample(1).values.reshape(28, 28)

    

    ax1 = plt.subplot(gs1[i])

    ax1.set_xticks([])

    ax1.set_yticks([])

    ax1.imshow(a, cmap=plt.cm.binary)
X_train = tf.keras.utils.normalize(X_train, axis=1)

X_test = tf.keras.utils.normalize(X_test, axis=1)
avg_values = X_train.mean().values.reshape(28, 28)



plt.imshow(avg_values, cmap=plt.cm.binary)

plt.title('Average Values of Training Data');
model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # relu is the go-to activation function for hidden layer

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))



model.compile(optimizer='adam', # adam is the go-to optimizer

                loss='sparse_categorical_crossentropy',

                metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30)
val_loss, val_acc = model.evaluate(X_test, y_test)

print(val_loss, val_acc)
output = model.predict(X_test)

predictions = np.argmax(output, axis=1)
print(f'Predicted {predictions[0]}\nActual Value {y_test.head(1).values[0]}')

plt.imshow(X_test.head(1).values.reshape(28,28), cmap=plt.cm.binary);
kaggle_test = tf.keras.utils.normalize(kaggle_test, axis=1)
kaggle_output = model.predict(kaggle_test)

kaggle_predictions = np.argmax(kaggle_output, axis=1)
kaggle_predictions
kaggle_submission = pd.DataFrame({'ImageId': range(1,kaggle_test.shape[0]+1), 'Label': kaggle_predictions})
kaggle_submission.to_csv('Submission.csv', index=False)