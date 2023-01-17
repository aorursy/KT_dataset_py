import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential

%matplotlib inline
tf.enable_eager_execution()
tfe = tf.contrib.eager
import tensorflow.contrib.eager as tfe 
# def create_model():
#     model = Sequential()
#     model.add(Convolution2D(filters = 16, kernel_size = 3, padding = 'same', input_shape = [28, 28, 1], activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters = 256, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Convolution2D(filters = 512, kernel_size = 3, padding = 'same', activation = 'relu'))
#     model.add(MaxPooling2D(pool_size = (2,2)))
#     model.add(BatchNormalization())
#     model.add(Flatten())
#     model.add(Dense(units = 100, activation = 'relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(units = 10))
#     return model
def create_model():
    model = Sequential()
    model.add(Convolution2D(filters = 16, kernel_size = 3, padding = 'same', input_shape = [28, 28, 1]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters = 32, kernel_size = 3, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters = 64, kernel_size = 3, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters = 128, kernel_size = 3, padding = 'same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(units = 100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(units = 10))
    return model
model = create_model()
model.summary()
model(np.zeros((10, 28, 28, 1), np.float32))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
x_train = np.array(train.iloc[:,1:])
y_train = np.array(train.iloc[:,0])

x_test = np.array(test.iloc[:,:])
# y_test = np.array(test.iloc[:,0])
y_train = tf.one_hot(y_train, 10)
# y_test = tf.one_hot(y_test, 10)
batch_size = 256
dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).batch(batch_size)
dataset_iter = tfe.Iterator(dataset)
#evaluate the loss
def loss(model, x, y):
    prediction = model(x)
    return tf.losses.softmax_cross_entropy(y, logits=prediction)

#record the gradient with respect to the model variables 
def grad(model, x, y):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, y)
    return tape.gradient(loss_value, model.variables)

#calcuate the accuracy of the model 
def accuracy(model, x, y):

    #prediction
    yhat = model(x)

    #get the labels of the predicted values 
    yhat = tf.argmax(yhat, 1).numpy()

    #get the labels of the true values
    y    = tf.argmax(y   , 1).numpy()
    return np.sum(y == yhat)/len(y)
#use Adam optimizer 
optimizer = tf.train.AdamOptimizer()

#record epoch loss and accuracy  
loss_history = tfe.metrics.Mean("loss")
accuracy_history = tfe.metrics.Mean("accuracy")
def interval_mapping(image, from_min, from_max, to_min, to_max):
    # map values from [from_min, from_max] to [to_min, to_max]
    # image: input array
    from_range = from_max - from_min
    to_range = to_max - to_min
    scaled = np.array((image - from_min) / float(from_range), dtype=np.float32)
    return np.array(to_min + (scaled * to_range))
epoch = 0
epochs = 3000
while epoch < epochs:
  #get next batch
    try:
        d = dataset_iter.next()
    except StopIteration:
        dataset_iter = tfe.Iterator(dataset)
        d = dataset_iter.next()
    
    # Images
    x_batch = d[0]
    x_batch = tf.reshape(x_batch, shape=[-1, 28, 28, 1])
    x_batch = interval_mapping(x_batch, 0, 1, -1, 1)
    x = tf.convert_to_tensor(x_batch)
    # Labels
    y = d[1]
    
    

    # Calculate derivatives of the input function with respect to its parameters.
    grads = grad(model, x, y)

    # Apply the gradient to the model
    optimizer.apply_gradients(zip(grads, model.variables),
                            global_step=tf.train.get_or_create_global_step())
  
    #record the current loss and accuracy   
    loss_history(loss(model, x, y))
    accuracy_history(accuracy(model, x, y))
  
    if epoch % 100 == 0:
        print("epoch: {:d} Loss: {:.3f}, Acc: {:.3f}".format(epoch, loss_history.result(), accuracy_history.result()))

    #clear the history 
    loss_history.init_variables()
    accuracy_history.init_variables()

    epoch += 1
x_test = np.array(tf.reshape(x_test, shape=[-1, 28, 28, 1]))
x_test_im = np.array(tf.reshape(x_test, shape=[-1, 28, 28]))
x_test = interval_mapping(x_test, 0, 1, -1, 1)
x_test = tf.convert_to_tensor(x_test)
pred_test = tf.argmax(model(x_test).numpy(), axis=1)
print(pred_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred_test.numpy())+1)),
                         "Label": pred_test.numpy()})
submissions.to_csv("submission_3.csv", index=False, header=True)
op = pd.read_csv("submission_3.csv")
print(op)
