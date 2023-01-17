import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D

from tensorflow.keras import Model

from tensorflow.keras.models import Sequential



from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/digit-recognizer/train.csv")

test_df = pd.read_csv("../input/digit-recognizer/test.csv")

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

sample_submission.sample()
y_train, x_train = train_df.iloc[:,0].values, train_df.iloc[:,1:].values
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 42)

x_train, x_test = x_train / 255.0, x_test / 255.0



# Add a channels dimension

x_train = x_train[..., tf.newaxis]

x_test = x_test[..., tf.newaxis]

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

x_train = x_train.reshape((x_train.shape[0],28,28)).astype('float32')

x_test = x_test.reshape((x_test.shape[0],28,28)).astype('float32')
# Add a channels dimension

x_train = x_train[..., tf.newaxis]

x_test = x_test[..., tf.newaxis]
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

train_ds = tf.data.Dataset.from_tensor_slices(

    (x_train, y_train)).shuffle(10000).batch(32)



test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class MyModel(Model):

    def __init__(self):

        super(MyModel, self).__init__()

        self.conv1 = Conv2D(32, 3, activation='relu')

        self.flatten = Flatten()

        self.d1 = Dense(128, activation='relu')

        self.d2 = Dense(10, activation='softmax')



    def call(self, x):

        x = self.conv1(x)

        x = self.flatten(x)

        x = self.d1(x)

        return self.d2(x)



# Create an instance of the model

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()



optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')

train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')



test_loss = tf.keras.metrics.Mean(name='test_loss')

test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
@tf.function

def train_step(images, labels):

    with tf.GradientTape() as tape:

        predictions = model(images)

        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



    train_loss(loss)

    train_accuracy(labels, predictions)
@tf.function

def test_step(images, labels):

    predictions = model(images)

    t_loss = loss_object(labels, predictions)



    test_loss(t_loss)

    test_accuracy(labels, predictions)
EPOCHS = 10



for epoch in range(EPOCHS):

    for images, labels in train_ds:

        train_step(images, labels)



    for test_images, test_labels in test_ds:

        test_step(test_images, test_labels)



    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'

    print(template.format(epoch+1,

                        train_loss.result(),

                        train_accuracy.result()*100,

                        test_loss.result(),

                        test_accuracy.result()*100))



    # Reset the metrics for the next epoch

    train_loss.reset_states()

    train_accuracy.reset_states()

    test_loss.reset_states()

    test_accuracy.reset_states()
test_df.head()
test_inp = test_df.to_numpy()

test_inp.shape

test_inp = test_inp[..., tf.newaxis]

test_inp = test_inp.reshape((test_inp.shape[0],28,28)).astype('float32')

test_inp = test_inp[..., tf.newaxis]

test_inp.shape
predictions = model(test_inp)
type(predictions)
predictions.shape
pred = predictions.numpy()

pred.shape
predictions = np.argmax(pred, axis = 1)

predictions.shape
image_id = np.arange(1, len(test_df) + 1)

image_id.shape
columns = sample_submission.columns
submission = pd.DataFrame({

    columns[0]: image_id,

    columns[1]:predictions

})

submission.head()
submission.to_csv("submission.csv", index = False)