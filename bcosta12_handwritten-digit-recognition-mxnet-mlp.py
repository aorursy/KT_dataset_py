import mxnet as mx

import pandas as pd

import numpy as np

import logging

from sklearn.model_selection import train_test_split



logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

path = '../input/'
# Fix the seed

mx.random.seed(7)



# Set the compute context, GPU is available otherwise CPU

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
print("model extract train init")

df_train = pd.read_csv(path + 'train.csv')

y = (np.array(df_train['label'].values.tolist()).astype(np.int)).copy()

df_train = df_train.drop(columns=['label'])

X = (np.array(df_train.values.tolist()).astype(np.float)).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("model extract train end")
print("model extract test init")

df_test_final = pd.read_csv(path + 'test.csv')

X_test_final = np.array(df_test_final.values.tolist()).astype(np.float)

print("model extract train end")
batch_size = 100

train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)

val_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
data = mx.sym.var('data')

# Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)

data = mx.sym.flatten(data=data)
# The first fully-connected layer and the corresponding activation function

fc1  = mx.sym.FullyConnected(data=data, num_hidden=128)

act1 = mx.sym.Activation(data=fc1, act_type="relu")



# The second fully-connected layer and the corresponding activation function

fc2  = mx.sym.FullyConnected(data=act1, num_hidden = 64)

act2 = mx.sym.Activation(data=fc2, act_type="relu")
# MNIST has 10 classes

fc3  = mx.sym.FullyConnected(data=act2, num_hidden=10)

# Softmax with cross entropy loss

mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')
mlp_model = mx.mod.Module(symbol=mlp, context=ctx)

mlp_model.fit(train_iter,  # train data

              eval_data=val_iter,  # validation data

              optimizer='sgd',  # use SGD to train

              optimizer_params={'learning_rate':0.01},  # use fixed learning rate

              eval_metric='acc',  # report accuracy during training

              batch_end_callback = mx.callback.Speedometer(batch_size, 50), # output progress for each 100 data batches

              num_epoch=15)  # train for at most 10 dataset passes
test_iter = mx.io.NDArrayIter(X_test_final, None, batch_size)

prob = mlp_model.predict(test_iter)

assert prob.shape == (len(X_test_final), 10)



y = []

for p in list(prob):

    y.append(list(p).index(np.max(p)))
test_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)

# predict accuracy of mlp

acc = mx.metric.Accuracy()

mlp_model.score(test_iter, acc)

print(acc)

assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]
df = pd.DataFrame({'ImageId': [x for x in range(1, len(y) +1 )], 'Label': y})

df.to_csv('submission.csv', index=False)