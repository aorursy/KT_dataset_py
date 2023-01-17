from mxnet import gluon
from mxnet import ndarray as nd

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
gluon.data.vision.FashionMNIST
# mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
# mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
import pandas as pd
train = pd.read_csv('../input/fashion-mnist_train.csv')
test = pd.read_csv('../input/fashion-mnist_test.csv')

### load data

train_y = train['label'].values
train_x = train.drop(columns=['label'])
train_x = train_x.values.reshape(-1,28,28,1)
print(train_x.shape,test.shape)
test_y = test['label'].values
test = test.drop(columns=['label'])
test_x = test.values.reshape(-1,28,28,1)
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')
train_x /= 255.0
test_x /= 255.0
mnist_train = gluon.data.ArrayDataset(nd.array(train_x), nd.array(train_y))
mnist_test = gluon.data.ArrayDataset(nd.array(test_x), nd.array(test_y))
import matplotlib.pyplot as plt

def show_images(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

data, label = mnist_train[0:9]
show_images(data)
print(get_text_labels(label))
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
### init
numinput = 28*28
numoutput = 10
w = nd.random_normal(shape=(numinput, numoutput))
b = nd.random_normal(shape=(numoutput,))
params = [w,b]
# attach grad
for param in params:
    param.attach_grad()
def softmax(X):
    X = X  - nd.max(X) #handle overflow and unflow
    exp = nd.exp(X)
    s = exp.sum(axis=1,keepdims=True)
    return exp/s
###test

X = nd.random_normal(shape=(2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(axis=1))
# define models
def net(X):
    return softmax(nd.dot(X.reshape((-1,numinput)),w)+b)
def cross_entropy(yhat, y):
    return - nd.pick(nd.log(yhat), y)
    
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

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


from mxnet import autograd

learning_rate = 0.5#.1

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
#             print(output)
            loss = cross_entropy(output, label)
#         print(output)
        loss.backward()
        # 将梯度做平均，这样学习率会对batch size不那么敏感
        SGD(params, learning_rate/batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))


data, label = mnist_test[0:9]
show_images(data)
print('true labels')
print(get_text_labels(label))

predicted_labels = net(data).argmax(axis=1)
print('predicted labels')
print(get_text_labels(predicted_labels.asnumpy()))
#################totally gluon
batch_size = 256
net1 = gluon.nn.Sequential()
with net1.name_scope():
    net1.add(gluon.nn.Flatten())
    net1.add(gluon.nn.Dense(10))
net1.initialize()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net1.collect_params(),'sgd',{'learning_rate':0.1})
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net1(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))

