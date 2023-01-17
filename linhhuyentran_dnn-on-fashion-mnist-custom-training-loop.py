# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import models

from tensorflow.keras import layers

from tensorflow.keras import optimizers

from keras.regularizers import l1, l2



# Helper libraries

from datetime import datetime

import os

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical
#reload the dataset into train and test

train, test = tf.keras.datasets.fashion_mnist.load_data()



#split train, validation sets

x, y = train

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)



x_train = x_train.reshape(len(x_train), 28 * 28).astype('float32')/255

y_train = to_categorical(y_train)



x_valid = x_valid.reshape(len(x_valid), 28 * 28).astype('float32')/255

y_valid = to_categorical(y_valid)



#process test set

x_test, y_test = test

x_test = x_test.reshape(10000, 28 * 28).astype('float32')/255



#construct pipelines

train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))

valid_set = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))

test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))



#batch the sets according to batch_size

batch_size = 64

train_batches = train_set.shuffle(buffer_size=1024).batch(batch_size)

valid_batches = valid_set.batch(batch_size)

test_batches = test_set.batch(batch_size)
#baseline model:

#batch the sets according to batch_size

model = models.Sequential()

model.add(layers.Dense(256, activation='relu', kernel_initializer = 'he_normal', input_shape = (28*28,)))

model.add(layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))

model.add(layers.Dense(64, activation='relu', kernel_initializer = 'he_normal'))

model.add(layers.Dense(10, activation='softmax'))

# Instantiate an optimizer

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=2e-06, nesterov=True)
keras.utils.plot_model(model, show_shapes=True)
# Instantiate a loss function

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

# Prepare the metrics

train_acc_metric = keras.metrics.CategoricalAccuracy()

train_loss_avg = tf.keras.metrics.Mean()

train_norm_avg = tf.keras.metrics.Mean()

val_acc_metric = keras.metrics.CategoricalAccuracy()

val_loss_avg = tf.keras.metrics.Mean()



# Use the tf.GradientTape context to calculate the gradients used to optimize model:

def grad(model, x, y):

  with tf.GradientTape() as tape:

    probs = model(x)

    loss_value = loss_fn(y, probs)

    grads = tape.gradient(loss_value, model.trainable_variables)

    norms = [tf.norm(each) for each in grads]

  return loss_value, grads, norms



num_epochs = 100
# Keep results for plotting

train_accuracy_results = []

train_loss_results = []

train_grad_results = []

train_norm_results = []

valid_accuracy_results = []

valid_loss_results = []

learning_rate_records = []



for epoch in range(num_epochs):

  # Iterate over the batches of train dataset.

  for x_batch_train, y_batch_train in train_batches:

    loss_value, grads, norms = grad(model, x_batch_train, y_batch_train)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    lr = optimizer._decayed_lr(var_dtype=tf.float32)

    # Track progress

    train_acc_metric.update_state(y_batch_train, model(x_batch_train)) # Add current batch acc

    train_loss_avg.update_state(loss_value) # Add current batch loss

    train_norm_avg.update_state(norms)  # Add current batch norm



  # Run a validation loop over the batches of validation dataset.

  for x_batch_val, y_batch_val in valid_batches:

    val_loss, _, _ = grad(model, x_batch_val, y_batch_val)

    #Track progress

    val_acc_metric.update_state(y_batch_val, model(x_batch_val)) # Add current batch acc

    val_loss_avg.update_state(val_loss) # Add current batch loss

  

  # End epoch

  train_accuracy_results.append(train_acc_metric.result())

  train_loss_results.append(train_loss_avg.result())

  train_grad_results.append(grads)

  train_norm_results.append(train_norm_avg.result())

  valid_accuracy_results.append(val_acc_metric.result())

  valid_loss_results.append(val_loss_avg.result())

  learning_rate_records.append(lr)

  #print to console

  if epoch % 1 == 0:

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Val-Loss: {:.3f}, Val-Accuracy: {:.3%}, L2 norm: {:.3f}, Learning rate: {}".format(epoch+1,

                                                                train_loss_avg.result(),

                                                                train_acc_metric.result(),

                                                                val_loss_avg.result(),

                                                                val_acc_metric.result(),

                                                                train_norm_avg.result(),

                                                                lr))



#evaluate on test set

test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

for (x, y) in test_batches:

  probs = model(x)

  prediction = tf.argmax(probs, axis=1, output_type=tf.int32)

  test_acc_metric(prediction, y)

print("Test-Accuracy: {:.3%}".format(test_acc_metric.result()))
# #log and display gradients and their l2 norms into tensorboard

# writer = tf.summary.create_file_writer("./logs/gradient")

# with writer.as_default():

#   with tf.name_scope("gradient"):

#     for epoch, grads in zip(range(num_epochs), train_grad_results):

#       for weight, grad in zip(model.weights, grads):

#         tf.summary.histogram('{}_grad'.format(weight.name), grad, step = epoch)

#   with tf.name_scope("gradient_norm"):

#     for epoch, norm in zip(range(num_epochs), train_norm_results):

#       tf.summary.scalar(name = "gradient_norm", data = norm, step = epoch)

# writer.flush()



# %reload_ext tensorboard

# %tensorboard --logdir=logs



# #log and display train-validation accuracy, loss and learning rate into tensorboard

# root_logdir = "logs"

# run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

# logdir = os.path.join(root_logdir, run_id)

# !rm -rf ./logs



# writer1 = tf.summary.create_file_writer("./logs/train")

# with writer1.as_default():

#   with tf.name_scope("loss"):

#     for epoch, train_loss in zip(range(num_epochs), train_loss_results):

#       tf.summary.scalar(name = "loss", data = train_loss, step = epoch)

#   with tf.name_scope("accuracy"):

#     for epoch, train_acc in zip(range(num_epochs), train_accuracy_results):

#       tf.summary.scalar(name = "accuracy", data = train_acc, step = epoch)

# writer1.flush()



# writer2 = tf.summary.create_file_writer("./logs/validation")

# with writer2.as_default():

#   with tf.name_scope("loss"): 

#     for epoch, val_loss in zip(range(num_epochs), valid_loss_results):

#       tf.summary.scalar(name = "loss", data = val_loss, step = epoch)

#   with tf.name_scope("accuracy"):

#     for epoch, val_acc in zip(range(num_epochs), valid_accuracy_results):

#       tf.summary.scalar(name = "accuracy", data = val_acc, step = epoch)

# writer2.flush()



# writer3 = tf.summary.create_file_writer("./logs/lr")

# with writer3.as_default():

#   with tf.name_scope("learning rate"):

#     for epoch, lr in zip(range(num_epochs), learning_rate_records):

#       tf.summary.scalar(name = "lr", data = lr, step = epoch)

# writer3.flush()
#dropout model:

model = models.Sequential()

model.add(layers.Dense(256, activation='relu', kernel_initializer = 'he_normal', input_shape = (28*28,)))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(128, activation='relu', kernel_initializer = 'he_normal'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(64, activation='relu', kernel_initializer = 'he_normal'))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(10, activation='softmax'))

# Instantiate an optimizer

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=2e-06, nesterov=True)
keras.utils.plot_model(model, show_shapes=True)
# Instantiate a loss function

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

# Prepare the metrics

train_acc_metric = keras.metrics.CategoricalAccuracy()

train_loss_avg = tf.keras.metrics.Mean()

train_norm_avg = tf.keras.metrics.Mean()

val_acc_metric = keras.metrics.CategoricalAccuracy()

val_loss_avg = tf.keras.metrics.Mean()



# Use the tf.GradientTape context to calculate the gradients used to optimize model:

def grad(model, x, y):

  with tf.GradientTape() as tape:

    probs = model(x, training = True)

    # training=True is needed only if there are layers with different

    # behavior during training versus inference (e.g. Dropout).

    loss_value = loss_fn(y, probs)

    grads = tape.gradient(loss_value, model.trainable_variables)

    norms = [tf.norm(each) for each in grads]

  return loss_value, grads, norms



num_epochs = 100
# Keep results for plotting

re_train_accuracy_results = []

re_train_loss_results = []

re_train_grad_results = []

re_train_norm_results = []

re_valid_accuracy_results = []

re_valid_loss_results = []

re_learning_rate_records = []



for epoch in range(num_epochs):

  # Iterate over the batches of train dataset.

  for x_batch_train, y_batch_train in train_batches:

    loss_value, grads, norms = grad(model, x_batch_train, y_batch_train)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    lr = optimizer._decayed_lr(var_dtype=tf.float32)

    weights = model.trainable_variables

    # Track progress

    train_acc_metric.update_state(y_batch_train, model(x_batch_train, training = True)) # Add current batch acc

    # training=True is needed only if there are layers with different

    # behavior during training versus inference (e.g. Dropout).

    train_loss_avg.update_state(loss_value) # Add current batch loss

    train_norm_avg.update_state(norms)  # Add current batch norm



  # Run a validation loop over the batches of validation dataset.

  for x_batch_val, y_batch_val in valid_batches:

    val_loss, _, _ = grad(model, x_batch_val, y_batch_val)

    #Track progress

    val_acc_metric.update_state(y_batch_val, model(x_batch_val, training = True)) # Add current batch acc

    # training=True is needed only if there are layers with different

    # behavior during training versus inference (e.g. Dropout).

    val_loss_avg.update_state(val_loss) # Add current batch loss

  

  # End epoch

  re_train_accuracy_results.append(train_acc_metric.result())

  re_train_loss_results.append(train_loss_avg.result())

  re_train_grad_results.append(grads)

  re_train_norm_results.append(train_norm_avg.result())

  re_valid_accuracy_results.append(val_acc_metric.result())

  re_valid_loss_results.append(val_loss_avg.result())

  re_learning_rate_records.append(lr)

  #print to console

  if epoch % 1 == 0:

    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Val-Loss: {:.3f}, Val-Accuracy: {:.3%}, L2 norm: {:.3f}, Learning rate: {}".format(epoch+1,

                                                                train_loss_avg.result(),

                                                                train_acc_metric.result(),

                                                                val_loss_avg.result(),

                                                                val_acc_metric.result(),

                                                                train_norm_avg.result(),

                                                                lr))



#evaluate on test set

test_acc_metric = tf.keras.metrics.CategoricalAccuracy()

for (x, y) in test_batches:

  probs = model(x, training = False)

  # training=False is needed only if there are layers with different

  # behavior during training versus inference (e.g. Dropout).

  prediction = tf.argmax(probs, axis=1, output_type=tf.int32)

  test_acc_metric(prediction, y)

print("Test set accuracy: {:.3%}".format(test_acc_metric.result()))
import matplotlib.pyplot as plt



epochs = range(num_epochs)



# "bo" is for "blue dot"

plt.plot(epochs, train_loss_results, 'bo', label='Training loss')

plt.plot(epochs, re_train_loss_results, 'ro', label='Dropout Training loss')

# b is for "solid blue line"

plt.plot(epochs, valid_loss_results, 'b', label='Validation loss')

plt.plot(epochs, re_valid_loss_results, 'r', label='Dropout Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()



# "bo" is for "blue dot"

plt.plot(epochs, train_accuracy_results, 'bo', label='Training accuracy')

plt.plot(epochs, re_train_accuracy_results, 'ro', label='Dropout Training accuracy')

# b is for "solid blue line"

plt.plot(epochs, valid_accuracy_results, 'b', label='Validation accuracy')

plt.plot(epochs, re_valid_accuracy_results, 'r', label='Dropout Validation accuracy')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()