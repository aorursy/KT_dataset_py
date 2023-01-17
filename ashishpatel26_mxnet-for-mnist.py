import numpy as np
import pandas as pd
import mxnet as mx
import logging
# set to 0 to train on all available data
VALIDATION_SIZE = 2000
# create the training 
train = pd.read_csv("../input/train.csv")
target = train['label'].values.ravel()
train = train.iloc[:,1:].values
def get_lenet():
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet
# split dataset
val_data = train[:VALIDATION_SIZE].astype('float32')
val_label = target[:VALIDATION_SIZE]
train_data = train[VALIDATION_SIZE: , :].astype('float32')
train_label = target[VALIDATION_SIZE:]
train_data = np.array(train_data).reshape((-1, 1, 28, 28))
val_data = np.array(val_data).reshape((-1, 1, 28, 28))
# Normalize data
train_data[:] /= 256.0
val_data[:]/= 256.0
batch_size = 500
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)
# logging
head = '%(asctime)-15s Node[0] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
# create model 
devs = mx.gpu(0)
network=get_lenet()
model = mx.model.FeedForward(
        #ctx                = devs,
        symbol             = network,
        num_epoch          = 4,
        learning_rate      = 0.1,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34)
        )
eval_metrics = ['accuracy']
model.fit(
    X=train_iter, 
    eval_metric = eval_metrics,
    eval_data = val_iter
    )
#predict
test = pd.read_csv("../input/test.csv").values
test_data = test.astype('float32')
test_data = np.array(test_data).reshape((-1, 1, 28, 28))
test_data[:]/= 256.0
test_iter = mx.io.NDArrayIter(test_data, batch_size=batch_size)
pred = model.predict(X = test_iter)
pred = np.argsort(pred)
np.savetxt('mxnet_submission.csv', np.c_[range(1,len(test)+1),pred[:,9]], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')
