# -*- coding: utf-8 -*-

import chainer

import chainer.functions as F

import chainer.links as L

import numpy as np

import pandas as pd



class NeuralNet(chainer.Chain):

    def __init__(self):

        super(NeuralNet, self).__init__()

        with self.init_scope():

            self.layer1 = L.Linear(None, 10)

    def __call__(self, x):

        x = self.layer1(F.relu(x))

        return x



df = pd.read_csv('../input/train.csv')

X = df[df.columns[1:]].astype(np.float32).values

Y = df[df.columns[0]].values



nn = NeuralNet()

model = L.Classifier(nn)



train_iter = chainer.iterators.SerialIterator([(X[i],Y[i]) for i in range(len(X))], 200, shuffle=True)

optimizer = chainer.optimizers.AdaDelta()

optimizer.setup(model)

updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)

trainer = chainer.training.Trainer(updater, (5, 'epoch'), out="result")

trainer.extend(chainer.training.extensions.LogReport())

trainer.extend(chainer.training.extensions.PrintReport(['epoch','main/loss','main/accuracy']))

trainer.run()



df = pd.read_csv('../input/test.csv')

df.head()

result = nn(df.astype(np.float32).values)

result = [np.argmax(x) for x in result.data]

df = pd.DataFrame({'ImageId': range(1,len(result)+1),'Label': result})

df.to_csv('submission.csv', index=False)