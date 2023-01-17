!pip3 install ipywidgets

!jupyter nbextension enable --py --sys-prefix widgetsnbextension

!pip install torchtext==0.2.3

!pip install Pillow==4.1.1

!pip install blosc

## Fastai 0.7 with minor changes has been installed from personal Gitub repo : https://github.com/pritesh2312/fastai-v0.7
%load_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split



from fastai.imports import *

from fastai.torch_imports import *

from fastai.io import *



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
df_trn = train.drop(['label'], axis = 1)

y_trn =  train['label']



x_train, x_val, y_train, y_val = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)
x_train = x_train.values

type(x_train), x_train.shape
y_train = y_train.values

type(y_train), y_train.shape
x_val = x_val.values

type(x_val), x_val.shape
y_val = y_val.values

type(y_val), y_val.shape
x_test = test.values
mean = x_train.mean()

std = x_train.std()



x_train = (x_train-mean)/std

mean, std, x_train.mean(), x_train.std()
x_val = (x_val-mean)/std

x_val.mean(), x_val.std()
x_test = (x_test-mean)/std

x_test.mean(), x_test.std()
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
x_imgs = np.reshape(x_val, (-1,28,28)); x_imgs.shape
show(x_imgs[0], y_val[0])
plots(x_imgs[:8], titles=y_val[:8])
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

#).cuda()  ## For GPU

)         ## For CPU
loss=nn.NLLLoss()

metrics=[accuracy]

# opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9)

opt=optim.SGD(net.parameters(), 1e-1, momentum=0.9, weight_decay=1e-3)
#?ImageClassifierData.from_arrays
md = ImageClassifierData.from_arrays('../input/', (x_train,y_train), (x_val, y_val), test = x_test)
fit(net, md, n_epochs=5, crit=loss, opt=opt, metrics=metrics)
set_lrs(opt, 1e-2)
fit(net, md, n_epochs=10, crit=loss, opt=opt, metrics=metrics)
preds = predict(net, md.val_dl)

preds.shape
preds = preds.argmax(1)
np.mean(preds == y_val)
plots(x_imgs[:8], titles=preds[:8])
preds = predict(net, md.test_dl)

preds.shape
#??predict(net, )
preds = preds.argmax(1)
test_imgs = np.reshape(x_test, (-1,28,28)); test_imgs.shape
plots(test_imgs[:8], titles=preds[:8])
submit = pd.read_csv('../input/sample_submission.csv')

submit.head()
submit['Label'] = preds

submit.head()
submit.to_csv("pytorch_v1.csv", index=False)  ## To avoid adding the first column of row nos