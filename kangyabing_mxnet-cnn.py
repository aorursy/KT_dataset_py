%matplotlib inline

import mxnet as mx

import numpy as np

import matplotlib.pyplot as plt

import logging



logger = logging.getLogger()

logger.setLevel(logging.DEBUG)
# Variables are place holders for input arrays. We give each variable a unique name.

data = mx.symbol.Variable('data')



# The input is fed to a fully connected layer that computes Y=WX+b.

# This is the main computation module in the network.

# Each layer also needs an unique name. We'll talk more about naming in the next section.

fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)

# Activation layers apply a non-linear function on the previous layer's output.

# Here we use Rectified Linear Unit (ReLU) that computes Y = max(X, 0).

act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")



fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)

act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")



fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=10)

# Finally we have a loss layer that compares the network's output with label and generates gradient signals.

mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

mx.viz.plot_network(mlp)
import numpy as np

import pandas as pd

# load data to numpy



train_file ='../input/train.csv'

test_file = "../input/test.csv"

result_file = '../input/submission.csv'



df_train_data = pd.read_csv(train_file)

df_test_data = pd.read_csv(test_file)



label = np.array(df_train_data.pop('label'), dtype=int)

train_data = np.array(df_train_data.values, dtype=float)

test_data = np.array(df_test_data.values, dtype=float)
# plot data as grey image

for i in range(10):

    plt.subplot(1, 10, i + 1)

    plt.imshow(train_data[i].reshape(28, 28) * 255, cmap='Greys_r')

    plt.axis('off')

plt.show()
# sub mean value

mean_data = np.mean(train_data, axis=0)

X_train = (train_data - mean_data)/128

Y_train = label

X_test = (test_data - mean_data)/128



print(X_train.shape, X_test.shape)
batch_size = 128

train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)

test_iter = None
data = data.astype(np.float32) / 255

data_test = data_test.astype(np.float32) / 255
model = mx.model.FeedForward(

    ctx = mx.cpu(0),      # Run on GPU 0

    symbol = mlp,         # Use the network we just defined

    num_epoch = 15,       # Train for 10 epochs

    learning_rate = 0.1,  # Learning rate

    momentum = 0.9,       # Momentum for SGD with momentum

    wd = 0.00001)

model.fit(

    X=train_iter,  # Training data set

    eval_data=test_iter,  # Testing data set. MXNet computes scores on test set every epoch

    batch_end_callback = mx.callback.Speedometer(batch_size, 100))  # Logging module to print out progress
# Predict the test data

result = []

for i in range(X_test.shape[0]):

    label = model.predict(X_test[i:i+1])[0].argmax()

    result.append(label)
submission = pd.DataFrame(data={'ImageId':(np.arange(len(result)) + 1), 'Label':np.array(result)})

submission.to_csv('./submission.csv', index=False)

submission.tail()