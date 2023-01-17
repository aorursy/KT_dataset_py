from mxnet import gluon
from mxnet import ndarray as nd

# mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
# mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
import pandas as pd
def getfashionmist(batch_size):
    train = pd.read_csv('../input/fashion-mnist_train.csv')
    test = pd.read_csv('../input/fashion-mnist_test.csv')
    ## load data

    train_y = train['label'].values
    train_x = train.drop(columns=['label'])
    train_x = train_x.values.reshape(-1,28,28,1)
    test_y = test['label'].values
    test = test.drop(columns=['label'])
    test_x = test.values.reshape(-1,28,28,1)
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255.0
    test_x /= 255.0
    mnist_train = gluon.data.ArrayDataset(nd.array(train_x), nd.array(train_y))
    mnist_test = gluon.data.ArrayDataset(nd.array(test_x), nd.array(test_y))
    
    batch_size = 256
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    return train_data,test_data
batch_size = 256
train_data,test_data = getfashionmist(batch_size)
num_inputs = 28*28
num_ouputs = 10

num_hidden = 256
weight_scale = 0.01
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)

W2 = nd.random_normal(shape=(num_hidden,num_ouputs),scale=weight_scale)
b2 = nd.zeros(num_ouputs)

params = [W1,b1,W2,b2]
for p in params:
    p.attach_grad()

def relu(X):
    return nd.maximum(0,X)
def net(X):
    X = X.reshape((-1,num_inputs))
    a1 = relu(nd.dot(X,W1)+b1)
    return nd.dot(a1,W2)+b2
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
def SGD(params, lr):
    for p in params:
        p[:] = p - lr*p.grad
    
def accuracy(output, label):
    return nd.mean(output.argmax(axis=1)==label).asscalar()
def evaluate_accuracy(data_iterator, net):
    acc = 0.
    for data, label in data_iterator:
        output = net(data)
#         print(output,1111,label)
        acc += accuracy(output, label)
    return acc / len(data_iterator)
evaluate_accuracy(test_data, net)
from mxnet import autograd
learning_rate = 0.5
for epoch in range(5):
    train_loss =0
    train_acc = 0
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output,label)
        loss.backward()
        SGD(params,learning_rate/batch_size)
        
        train_loss += loss.mean().asscalar()
        train_acc += accuracy(output,label)
    
    test_acc = evaluate_accuracy(test_data,net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
    epoch, train_loss/len(train_data),
    train_acc/len(train_data), test_acc))
#######gluon
from mxnet import gluon
net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Dense(256,activation='relu'))
    net1.add(gluon.nn.Dense(10))

net1.initialize()
def train_net(net1,lr = 0.5, epochs=5):
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net1.collect_params(), 'sgd', {'learning_rate': lr})
    for epoch in range(epochs):
        train_loss =0
        train_acc = 0
        for data, label in train_data:
            with autograd.record():
                output = net1(data)
                loss = softmax_cross_entropy(output,label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += loss.mean().asscalar()
            train_acc += accuracy(output,label)

        test_acc = evaluate_accuracy(test_data,net1)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
train_net(net1)
#####more layer
net2 = gluon.nn.Sequential()
with net2.name_scope():
    net2.add(gluon.nn.Dense(256,activation='relu'))
    net2.add(gluon.nn.Dense(128,activation='relu'))
    net2.add(gluon.nn.Dense(64,activation='relu'))
    net2.add(gluon.nn.Dense(10))

net2.initialize()
train_net(net2,lr=0.1,epochs=10)
help(nd.Activation)

















