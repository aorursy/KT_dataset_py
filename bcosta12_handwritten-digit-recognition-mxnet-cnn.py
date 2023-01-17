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
def reshare_array(array, dim):

    return np.reshape(array, (-1, 1, dim, dim))
X_train = reshare_array(X_train, 28)

X_test = reshare_array(X_test, 28)

X_test_final = reshare_array(X_test_final, 28)

y_train = np.array(y_train)

y_test = np.array(y_test)
batch_size = 100

train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size)

val_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)
data = mx.sym.var('data')

# first conv layer

conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)

tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")

pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

# second conv layer

conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)

tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")

pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))

# first fullc layer

flatten = mx.sym.flatten(data=pool2)

fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)

tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")

# second fullc

fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)

# softmax loss

lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
lenet_model = mx.mod.Module(symbol=lenet, context=ctx)
# train with the same

lenet_model.fit(train_iter,

                eval_data=val_iter,

                optimizer='sgd',

                optimizer_params={'learning_rate':0.005},

                eval_metric='acc',

                batch_end_callback = mx.callback.Speedometer(batch_size, 100),

                num_epoch=150)
test_iter = mx.io.NDArrayIter(X_test_final, None, batch_size)

prob = lenet_model.predict(test_iter)



y = []

for p in list(prob):

   y.append(list(p).index(np.max(p)))
test_iter = mx.io.NDArrayIter(X_test, y_test, batch_size)

# predict accuracy of mlp

acc = mx.metric.Accuracy()

lenet_model.score(test_iter, acc)

print(acc)

assert acc.get()[1] > 0.96, "Achieved accuracy (%f) is lower than expected (0.96)" % acc.get()[1]
df = pd.DataFrame({'ImageId': [x for x in range(1, len(y) + 1)], 'Label': y})

df.to_csv('submission.csv', index=False)