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
import chainer

import chainer.functions as F

import chainer.links as L

import numpy as np

import pandas as pd



class NeuralNet(chainer.Chain):

    def __init__(self):

        super(NeuralNet, self).__init__()

        with self.init_scope():

#             self.conv1 = L.Convolution2D(1, 1, 3,stride=1, pad=2,dilate=2)

#             self.conv2 = L.Convolution2D(1, 1, 3,stride=1, pad=2,dilate=2)

#             self.conv3 = L.Convolution2D(1, 1, 3,stride=1, pad=2,dilate=2)

#             self.conv4 = L.Convolution2D(1, 1, 3,stride=1, pad=2,dilate=2)

#             self.conv5 = L.Convolution2D(1, 1, 3,stride=1, pad=2,dilate=2)

#             self.conv1 = L.Convolution2D(1, 1, 5,stride=1, pad=0,dilate=1)

#             self.conv2 = L.Convolution2D(1, 1, 5,stride=1, pad=0,dilate=1)

#             self.layer2 = L.Linear(784, 10)



#             self.conv1=L.Convolution2D(1, 32, 5)

#             self.conv2=L.Convolution2D(32, 64, 5)

#             self.l1=L.Linear(1024, 10)



            self.conv1=L.Convolution2D(1, 32, 3,stride=1, pad=0,dilate=2)

            self.conv2=L.Convolution2D(32, 64, 3,stride=1, pad=0,dilate=2)

            self.l1=L.Linear(1024, 10)

            

    def __call__(self, x):

#         h1 = (F.relu(self.conv1(x))   + x)

#         h2 = (F.relu(self.conv2(h1))   + h1)

#         h3 = (F.relu(self.conv3(h2))   + h2)

#         h4 = (F.relu(self.conv4(h3))   + h3)

#         h5 = (F.relu(self.conv5(h4))   + h4)

# #         h1 = F.relu(self.conv1(x))

# #         h2 = F.relu(self.conv2(h1))

#         x = self.layer2(F.relu(h5))

#         return x

        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)

        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)

        return self.l1(h2)



df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X = df[df.columns[1:]].astype(np.float32).values

Y = df[df.columns[0]].values

X = X / 255.0

X = X.reshape(-1,1,28,28)

Y = df[df.columns[0]].values



nn = NeuralNet()

model = L.Classifier(nn)



train_iter = chainer.iterators.SerialIterator([(X[i],Y[i]) for i in range(len(X))], 64, shuffle=True)

optimizer = chainer.optimizers.AdaDelta()

optimizer.setup(model)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)

trainer = chainer.training.Trainer(updater, (10, 'epoch'), out="result")

trainer.extend(chainer.training.extensions.LogReport())

trainer.extend(chainer.training.extensions.PrintReport(['epoch','main/loss','main/accuracy']))

trainer.run()



df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

X_t = df[df.columns[0:]].astype(np.float32).values

X_t = X_t / 255.0

X_t = X_t.reshape(-1,1,28,28)

result = nn(X_t)

result = [np.argmax(x) for x in result.data]

df = pd.DataFrame({'ImageId': range(1,len(result)+1),'Label': result})

df.to_csv('submission_3.csv', index=False)

print("end")