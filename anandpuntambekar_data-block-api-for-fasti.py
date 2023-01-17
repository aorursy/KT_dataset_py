# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *
data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)

df_train.head(5)
df_train['label']=df_train['label'].replace(0, "T-shirt-top")

df_train['label']=df_train['label'].replace(1, "Trouser")

df_train['label']=df_train['label'].replace(2, "Pullover")

df_train['label']=df_train['label'].replace(3, "Dress")

df_train['label']=df_train['label'].replace(4, "Coat")

df_train['label']=df_train['label'].replace(5, "Sandal")

df_train['label']=df_train['label'].replace(6, "Shirt")

df_train['label']=df_train['label'].replace(7, "Sneaker")

df_train['label']=df_train['label'].replace(8, "Bag")

df_train['label']=df_train['label'].replace(9, "Ankle, boot")



df_test['label']=df_test['label'].replace(0, "T-shirt-top")

df_test['label']=df_test['label'].replace(1, "Trouser")

df_test['label']=df_test['label'].replace(2, "Pullover")

df_test['label']=df_test['label'].replace(3, "Dress")

df_test['label']=df_test['label'].replace(4, "Coat")

df_test['label']=df_test['label'].replace(5, "Sandal")

df_test['label']=df_test['label'].replace(6, "Shirt")

df_test['label']=df_test['label'].replace(7, "Sneaker")

df_test['label']=df_test['label'].replace(8, "Bag")

df_test['label']=df_test['label'].replace(9, "Ankle, boot")

# Select all columns but the first

x_train = df_train[df_train.columns[1:]]

x_test = df_test[df_test.columns[1:]]



x_train.head(2)
y_train = df_train['label']

y_test = df_test['label']



y_train.head(2)


# PyTorch puts channel first, so we will reshape the images as one channel 28x28

print(x_train.shape)

x_train = np.asarray(x_train.iloc[:,:]).reshape([ 60000,1,28,28])

print(x_train.shape)


# PyTorch puts channel first, so we will reshape the images as one channel 28x28

print(x_test.shape)

x_test = np.asarray(x_test.iloc[:,:]).reshape([ 10000,1,28,28])

print(x_test.shape)

import os



path = "./data"

os.mkdir(path)



path = "./data/train"

os.mkdir(path)



path = "./data/test"

os.mkdir(path)



lst=[ "Trouser",

"Pullover",

"Dress",

"Coat",

"Sandal",

"Shirt",

"Sneaker",

"Bag",

"Ankle, boot",

"T-shirt-top"]





for doc in lst:

    path = "./data/train/"+doc

    os.mkdir(path)



for doc in lst:

    path = "./data/test/"+doc

    os.mkdir(path)
import matplotlib

count=0

labels = df_train['label']

for L in labels:

    #print(L)

    matplotlib.image.imsave("./data/train/" +str(L)+"/" +str(count) +'_.png', x_train[count].squeeze())

    count=count+1
count=0

labels = df_test['label']

for L in labels:

    #print(L)

    matplotlib.image.imsave("./data/test/" +str(L)+"/" +str(count) +'_.png', x_test[count].squeeze())

    count=count+1
import os

os.listdir("./data/train/Coat") [:5]
path ="./data"

il = ImageList.from_folder(path, convert_mode='L')
#  Our image item list contains 70,000 items, and it's a bunch of images that are 1 by 28 by 28

#  PyTorch puts channel first, so they are one channel 28x28.



il.items.shape,il.items[0]
il[0].show()
# In this case, we want to use a binary color map. 

# In fast.ai, you can set a default color map. For more information about cmap and color maps, refer to the matplotlib documentation. 

# defaults.cmap='binary' world set the default color map for fast.ai.

defaults.cmap='binary'
defaults.cmap='binary'

il[0].show()
sd = il.split_by_folder(train='train', valid='test')

sd
ll = sd.label_from_folder()
ll
x,y = ll.train[500]
x.show()

print(y)
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])

ll = ll.transform(tfms)
# Create the Data Bunch

bs = 128

data = ll.databunch(bs=bs).normalize()
x,y = data.train_ds[-300]
x.show()

print(y)
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')

plot_multi(_plot, 3, 3, figsize=(8,8))
xb,yb = data.one_batch()

xb.shape,yb.shape
data.show_batch(rows=3, figsize=(8,8))
def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
model = nn.Sequential(

    conv(1, 8), # 14

    nn.BatchNorm2d(8),

    nn.ReLU(),

    conv(8, 16), # 7

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 32), # 4

    nn.BatchNorm2d(32),

    nn.ReLU(),

    conv(32, 16), # 2

    nn.BatchNorm2d(16),

    nn.ReLU(),

    conv(16, 10), # 1

    nn.BatchNorm2d(10),

    Flatten()     # remove (1,1) grid

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

print(learn.summary())
xb = xb.cuda()

model(xb).shape
xb = xb.cuda()

model(xb).shape

learn.lr_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.2)
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)
model = nn.Sequential(

    conv2(1, 8),   # 14

    conv2(8, 16),  # 7

    conv2(16, 32), # 4

    conv2(32, 16), # 2

    conv2(16, 10), # 1

    Flatten()      # remove (1,1) grid

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, max_lr=0.3)
class ResBlock(nn.Module):

    def __init__(self, nf):

        super().__init__()

        self.conv1 = conv_layer(nf,nf)

        self.conv2 = conv_layer(nf,nf)

        

    def forward(self, x): return x + self.conv2(self.conv1(x))
model = nn.Sequential(

    conv2(1, 8),

    res_block(8),

    conv2(8, 16),

    res_block(16),

    conv2(16, 32),

    res_block(32),

    conv2(32, 16),

    res_block(16),

    conv2(16, 10),

    Flatten()

)
def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))
model = nn.Sequential(

    conv_and_res(1, 8),

    conv_and_res(8, 16),

    conv_and_res(16, 32),

    conv_and_res(32, 16),

    conv2(16, 10),

    Flatten()

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
learn.lr_find(end_lr=100)

learn.recorder.plot()
learn.fit_one_cycle(12, max_lr=0.3)
#learn.lr_find(end_lr=100)

#learn.recorder.plot()