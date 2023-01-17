# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard

num_classes = 10

epochs = 5

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)
def cnn_model_fn(features, labels, mode):

  """Model function for CNN."""



  # Input Layer

    # Reshape X to 4-D tensor: [batch_size, width, height, channels]

  # MNIST images are 28x28 pixels, and have one color channel

  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])



  # Convolutional Layer #1

  # Computes 32 features using a 5x5 filter with ReLU activation.

  # Padding is added to preserve width and height.

  # Input Tensor Shape: [batch_size, 28, 28, 1]

  # Output Tensor Shape: [batch_size, 28, 28, 32]

  conv1 = tf.layers.conv2d(

      inputs=input_layer,

      filters=32,

      kernel_size=[5, 5],

      padding="same",

      activation=tf.nn.relu)



  # Pooling Layer #1

  # First max pooling layer with a 2x2 filter and stride of 2

  # Input Tensor Shape: [batch_size, 28, 28, 32]

  # Output Tensor Shape: [batch_size, 14, 14, 32]

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)



  # Convolutional Layer #2

  # Computes 64 features using a 5x5 filter.

  # Padding is added to preserve width and height.

  # Input Tensor Shape: [batch_size, 14, 14, 32]

  # Output Tensor Shape: [batch_size, 14, 14, 64]

  conv2 = tf.layers.conv2d(

      inputs=pool1,

      filters=64,

      kernel_size=[5, 5],

      padding="same",

      activation=tf.nn.relu)



  # Pooling Layer #2

  # Second max pooling layer with a 2x2 filter and stride of 2

  # Input Tensor Shape: [batch_size, 14, 14, 64]

  # Output Tensor Shape: [batch_size, 7, 7, 64]

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



  # Flatten tensor into a batch of vectors

  # Input Tensor Shape: [batch_size, 7, 7, 64]

  # Output Tensor Shape: [batch_size, 7 * 7 * 64]

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])



  # Dense Layer

  # Densely connected layer with 1024 neurons

  # Input Tensor Shape: [batch_size, 7 * 7 * 64]

  # Output Tensor Shape: [batch_size, 1024]

  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, name="dense1")



  # Add dropout operation; 0.6 probability that element will be kept

  dropout = tf.layers.dropout(

      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)



  # Logits layer

  # Input Tensor Shape: [batch_size, 1024]

  # Output Tensor Shape: [batch_size, 10]

  logits = tf.layers.dense(inputs=dropout, units=10)



  predictions = {

      # Generate predictions (for PREDICT and EVAL mode)

      "classes": tf.argmax(input=logits, axis=1),

      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the

      # `logging_hook`.

      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")

  }

  prediction_output = tf.estimator.export.PredictOutput({"classes": tf.argmax(input=logits, axis=1),

     "probabilities": tf.nn.softmax(logits, name="softmax_tensor")})



  if mode == tf.estimator.ModeKeys.PREDICT:

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,

        export_outputs={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_output})



  # Calculate Loss (for both TRAIN and EVAL modes)

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)

  loss = tf.losses.softmax_cross_entropy(

      onehot_labels=onehot_labels, logits=logits)

  # Generate some summary info

  tf.summary.scalar('loss', loss)

  tf.summary.histogram('conv1', conv1)

  tf.summary.histogram('dense', dense)



  # Configure the Training Op (for TRAIN mode)

  if mode == tf.estimator.ModeKeys.TRAIN:

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    train_op = optimizer.minimize(

        loss=loss,

        global_step=tf.train.get_global_step())



    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)



  # Add evaluation metrics (for EVAL mode)

  eval_metric_ops = {

      "accuracy": tf.metrics.accuracy(

          labels=labels, predictions=predictions["classes"])}

  return tf.estimator.EstimatorSpec(

      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
df_train[df_train.columns[1:]].values.shape
df_test.head()
train_data = np.array(df_train, dtype = 'float32')
test_data = np.array(df_test, dtype='float32')
x_train = train_data[:,1:]/255



y_train = train_data[:,0]



x_test= test_data[:,1:]/255



y_test=test_data[:,0]
x_train,x_validate,y_train,y_validate = train_test_split(x_train,y_train,test_size = 0.2,random_state = 12345)
image = x_train[55,:].reshape((28,28))

plt.imshow(image)

plt.show()

image_rows = 28



image_cols = 28



batch_size = 512



image_shape = (image_rows,image_cols,1) # Defined the shape of the image as 3d with rows and columns and 1 for the 3d visualisation
x_train = x_train.reshape(x_train.shape[0],*image_shape)

x_test = x_test.reshape(x_test.shape[0],*image_shape)

x_validate = x_validate.reshape(x_validate.shape[0],*image_shape)
cnn_model = Sequential([

    Conv2D(filters=32,kernel_size=3,activation='relu',input_shape = image_shape),

    MaxPooling2D(pool_size=2) ,# down sampling the output instead of 28*28 it is 14*14

    Dropout(0.2),

    Flatten(), # flatten out the layers

    Dense(32,activation='relu'),

    Dense(10,activation = 'softmax')

    

])
cnn_model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])
history = cnn_model.fit(

    x_train,

    y_train,

    batch_size=batch_size,

    epochs=50,

    verbose=1,

    validation_data=(x_validate,y_validate),

)
score = cnn_model.evaluate(x_test,y_test,verbose=0)

print('Test Loss : {:.4f}'.format(score[0]))

print('Test Accuracy : {:.4f}'.format(score[1]))
import matplotlib.pyplot as plt



%matplotlib inline



accuracy = history.history['accuracy']



val_accuracy = history.history['val_accuracy']



loss = history.history['loss']



val_loss = history.history['val_loss']



epochs = range(len(accuracy))



plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')



plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')



plt.title('Training and Validation accuracy')



plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training Loss')



plt.plot(epochs, val_loss, 'b', label='Validation Loss')



plt.title('Training and validation loss')



plt.legend()



plt.show()
#get the predictions for the test data



predicted_classes = cnn_model.predict_classes(x_test)



#get the indices to be plotted



y_true = df_test.iloc[:, 0]



correct = np.nonzero(predicted_classes==y_true)[0]



incorrect = np.nonzero(predicted_classes!=y_true)[0]



from sklearn.metrics import classification_report



target_names = ["Class {}".format(i) for i in range(num_classes)]



print(classification_report(y_true, predicted_classes, target_names=target_names))
for i, correct in enumerate(correct[:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_true[correct]))

    plt.tight_layout()
for i, incorrect in enumerate(incorrect[0:9]):

    plt.subplot(3,3,i+1)

    plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')

    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_true[incorrect]))

    plt.tight_layout()
