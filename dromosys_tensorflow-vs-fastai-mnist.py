import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.Dropout(0.2),
    nn.LogSoftmax()
).cuda()
path = 'data/mnist/'
import os
from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *
os.makedirs(path, exist_ok=True)

URL='http://deeplearning.net/data/mnist/'
FILENAME='mnist.pkl.gz'

def load_mnist(filename):
    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')
get_data(URL+FILENAME, path+FILENAME)
((x, y), (x_valid, y_valid), _) = load_mnist(path+FILENAME)
mean = x.mean()
std = x.std()

x=(x-mean)/std
mean, std, x.mean(), x.std()
x_valid = (x_valid-mean)/std
x_valid.mean(), x_valid.std()
md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid))
loss=nn.NLLLoss()
metrics=[accuracy]
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)
opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-3)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
preds = predict(net, md.val_dl)
preds.shape
preds = preds.argmax(1)
preds[:8]
def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)
x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape
show(x_imgs[0], y_valid[0])
y_valid[0]
show(x_imgs[0,10:15,10:15])
def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')
np.mean(preds == y_valid)
plots(x_imgs[:20], titles=preds[:20])


