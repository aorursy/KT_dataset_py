# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
FOLDERNAME = '/kaggle/input'

train_df = pd.read_csv(os.path.join(FOLDERNAME, 'digit-recognizer/train.csv'))

test_df = pd.read_csv(os.path.join(FOLDERNAME, 'digit-recognizer/test.csv'))
import tensorflow as tf

import matplotlib.pyplot as plt

from PIL import Image
train_df.head()
input_train = train_df.drop(columns=['label']).to_numpy()

target_train = train_df['label'].to_numpy()
def visualize_samples(input_arr, target_arr, num_classes=10):

    assert (num_classes > 0 and num_classes <= 10), 'Number of classes must be in range [1-10]'

    fig, axs = plt.subplots(1, num_classes, figsize=(1.5*num_classes, 2))

    if num_classes == 1:

        axs.imshow(input_arr[target_arr==0][0].reshape(28, 28), cmap='gray')

        axs.axis('off')

        axs.set_title('Digit 0')

    else:

        for i in range(num_classes):

            axs[i].imshow(input_arr[target_arr==i][0].reshape(28, 28), cmap='gray')

            axs[i].axis('off')

            axs[i].set_title('Digit {}'.format(i))
visualize_samples(input_train, target_train, 10)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(input_train, target_train, test_size=0.2, stratify=target_train)

X_train = X_train / 255.0

X_val = X_val / 255.0

# X_train = X_train.reshape(-1, 28, 28)

# X_val = X_val.reshape(-1, 28, 28)
# create Dataset object



train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
def train(model, optimizer, train_ds, val_ds, batch_size=32, num_epochs=1, print_every=1, is_training=False):

    train_ds = train_ds.batch(batch_size)

    val_ds = val_ds.batch(batch_size)

    

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

    

    val_loss = tf.keras.metrics.Mean(name='val_loss')

    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')

    

    for epoch in range(num_epochs):

        train_loss.reset_states()

        train_accuracy.reset_states()

        print('Epoch {}'.format(epoch+1))

        t = 0

        

        for X, y in train_ds:

            

            with tf.GradientTape() as tape:

                scores = model(X, training=is_training)

                loss = loss_fn(y, scores)

                gradients = tape.gradient(loss, model.trainable_variables)

                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                

                train_loss.update_state(loss)

                train_accuracy.update_state(y, scores)

                t += 1

            if t % print_every == 0:

                print('Iter {}, train loss: {:.4f}, train acc: {:.4f}'.format(t, train_loss.result(), train_accuracy.result()))

                      

        val_loss.reset_states()

        val_accuracy.reset_states()



        for Xval, yval in val_ds:

            scores_val = model(Xval, training=False)

            t_loss = loss_fn(yval, scores_val)



            val_loss.update_state(t_loss)

            val_accuracy.update_state(yval, scores_val)

        print('- train loss: {:.4f}, val loss: {:.4f}, train acc: {:.4f}, val acc: {:.4f}'.format(train_loss.result(), val_loss.result(),

                                                                                                train_accuracy.result(), val_accuracy.result()))

                

        print()
class Model(tf.keras.Model):

    def __init__(self, hidden_size1, hidden_size2, dropout_rate, num_classes):

        super(Model, self).__init__()

        self.fc1 = tf.keras.layers.Dense(hidden_size1)

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.relu1 = tf.keras.layers.ReLU()

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.fc2 = tf.keras.layers.Dense(hidden_size2)

        self.bn2 = tf.keras.layers.BatchNormalization()

        self.relu2 = tf.keras.layers.ReLU()

        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.fc_final = tf.keras.layers.Dense(num_classes, activation='softmax')

        

    def call(self, x, training=False):

        x = self.fc1(x)

#         x = self.bn1(x)

        x = self.relu1(x)

        x = self.dropout1(x)

        x = self.fc2(x)

#         x = self.bn2(x)

        x = self.relu2(x)

        x = self.dropout2(x)

        x = self.fc_final(x)

        return x

    

    

model = Model(500, 300, 0.5, 10)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=5000, decay_rate=.95)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
train(model, optimizer, train_ds, val_ds, 64, 10, 200, is_training=True)
input_test = test_df.to_numpy()

X_test = input_test / 255.0
test_ds = tf.data.Dataset.from_tensor_slices(input_test)
test_label = np.array([])

for test_batch in test_ds.batch(64):

    scores = model(test_batch, training=False)

    test_label = np.concatenate((test_label, np.argmax(scores, axis=1)))

    

test_label = test_label.astype(int)
output_df = pd.DataFrame({'ImageId': np.arange(1, len(test_label)+1), 'Label': test_label})
output_df.to_csv('output.csv', index=False)