%load_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.imports import *
from fastai.torch_imports import *
from fastai.io import *
path = 'data/mnist/'
import os
os.makedirs(path, exist_ok=True)
URL='http://deeplearning.net/data/mnist/'
FILENAME='mnist.pkl.gz'

def load_mnist(filename):
    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')
get_data(URL+FILENAME, path+FILENAME)
((x, y), (x_valid, y_valid), _) = load_mnist(path+FILENAME)
type(x), x.shape, type(y), y.shape
mean = x.mean()
std = x.std()

x=(x-mean)/std
mean, std, x.mean(), x.std()
x_valid = (x_valid-mean)/std
x_valid.mean(), x_valid.std()
def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)
def plots(ims, figsize=(12,6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')
x_valid.shape
x_imgs = np.reshape(x_valid, (-1,28,28)); x_imgs.shape
show(x_imgs[0], y_valid[0])
y_valid.shape
y_valid[0]
x_imgs[0,10:15,10:15]
show(x_imgs[0,10:15,10:15])
plots(x_imgs[:8], titles=y_valid[:8])
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *

import torch.nn as nn
net = nn.Sequential(
    nn.Linear(28*28, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.LogSoftmax()
).cuda()
md = ImageClassifierData.from_arrays(path, (x,y), (x_valid, y_valid))
loss=nn.NLLLoss()
metrics=[accuracy]
# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)
opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)
def binary_loss(y, p):
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))
acts = np.array([1, 0, 0, 1])
preds = np.array([0.9, 0.1, 0.2, 0.8])
binary_loss(acts, preds)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)

fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=3, crit=loss, opt=opt, metrics=metrics)
t = [o.numel() for o in net.parameters()]
t, sum(t)
preds = predict(net, md.val_dl)
preds.shape
preds.argmax(axis=1)[:5]
preds = preds.argmax(1)
np.mean(preds == y_valid)
plots(x_imgs[:8], titles=preds[:8])
def get_weights(*dims): return nn.Parameter(torch.randn(dims)/dims[0])
def softmax(x): return torch.exp(x)/(torch.exp(x).sum(dim=1)[:,None])

class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = (x @ self.l1_w) + self.l1_b  # Linear Layer
        x = torch.log(softmax(x)) # Non-linear (LogSoftmax) Layer
        return x
net2 = LogReg().cuda()
opt=optim.Adam(net2.parameters())
fit(net2, md, n_epochs=1, crit=loss, opt=opt, metrics=metrics)
dl = iter(md.trn_dl)
xmb,ymb = next(dl)
vxmb = Variable(xmb.cuda())
vxmb
preds = net2(vxmb).exp(); preds[:3]
preds = preds.data.max(1)[1]; preds
preds = predict(net2, md.val_dl).argmax(1)
plots(x_imgs[:8], titles=preds[:8])
np.mean(preds == y_valid)
a = np.array([10, 6, -4])
b = np.array([2, 8, 7])
a,b
a + b
(a < b).mean()
a
a > 0
a + 1
m = np.array([[1, 2, 3], [4,5,6], [7,8,9]]); m
2*m
c = np.array([10,20,30]); c
m + c
c + m
c.shape
np.broadcast_to(c[:,None], m.shape)
np.broadcast_to(np.expand_dims(c,0), (3,3))
c.shape
np.expand_dims(c,0).shape
np.expand_dims(c,0).shape
m + np.expand_dims(c,0)
np.expand_dims(c,1)
c[:, None].shape
m + np.expand_dims(c,1)
np.broadcast_to(np.expand_dims(c,1), (3,3))
c[None]
c[:,None]
c[None] > c[:,None]
xg,yg = np.ogrid[0:5, 0:5]; xg,yg
xg+yg
m, c
m @ c  # np.matmul(m, c)
T(m) @ T(c)
m,c
m * c
(m * c).sum(axis=1)
c
np.broadcast_to(c, (3,3))
n = np.array([[10,40],[20,0],[30,-5]]); n
m
m @ n
(m * n[:,0]).sum(axis=1)
(m * n[:,1]).sum(axis=1)
# Our code from above
class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x @ self.l1_w + self.l1_b 
        return torch.log(softmax(x))

net2 = LogReg().cuda()
opt=optim.Adam(net2.parameters())

fit(net2, md, n_epochs=1, crit=loss, opt=opt, metrics=metrics)
net2 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-3
optimizer=optim.Adam(net2.parameters(), lr=learning_rate)
dl = iter(md.trn_dl) # Data loader
xt, yt = next(dl)
y_pred = net2(Variable(xt).cuda())
l = loss(y_pred, Variable(yt).cuda())
print(l)
np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))
# Before the backward pass, use the optimizer object to zero all of the
# gradients for the variables it will update (which are the learnable weights
# of the model)
optimizer.zero_grad()

# Backward pass: compute gradient of the loss with respect to model parameters
l.backward()

# Calling the step function on an Optimizer makes an update to its parameters
optimizer.step()
xt, yt = next(dl)
y_pred = net2(Variable(xt).cuda())
l = loss(y_pred, Variable(yt).cuda())
print(l)
np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))
for t in range(100):
    xt, yt = next(dl)
    y_pred = net2(Variable(xt).cuda())
    l = loss(y_pred, Variable(yt).cuda())
    
    if t % 10 == 0:
        accuracy = np.mean(to_np(y_pred).argmax(axis=1) == to_np(yt))
        print("loss: ", l.data[0], "\t accuracy: ", accuracy)

    optimizer.zero_grad()
    l.backward()
    optimizer.step()
def score(x, y):
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)
net2 = LogReg().cuda()
loss=nn.NLLLoss()
learning_rate = 1e-2
optimizer=optim.SGD(net2.parameters(), lr=learning_rate)

for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        # Forward pass: compute predicted y and loss by passing x to the model.
        xt, yt = next(dl)
        y_pred = net2(V(xt))
        l = loss(y_pred, V(yt))
        losses.append(l)

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()

        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
    
    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))
net2 = LogReg().cuda()
loss_fn=nn.NLLLoss()
lr = 1e-2
w,b = net2.l1_w,net2.l1_b

for epoch in range(1):
    losses=[]
    dl = iter(md.trn_dl)
    for t in range(len(dl)):
        xt, yt = next(dl)
        y_pred = net2(V(xt))
        l = loss(y_pred, Variable(yt).cuda())
        losses.append(loss)

        # Backward pass: compute gradient of the loss with respect to model parameters
        l.backward()
        w.data -= w.grad.data * lr
        b.data -= b.grad.data * lr
        
        w.grad.data.zero_()
        b.grad.data.zero_()   

    val_dl = iter(md.val_dl)
    val_scores = [score(*next(val_dl)) for i in range(len(val_dl))]
    print(np.mean(val_scores))
