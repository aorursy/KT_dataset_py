# http://playground.tensorflow.org

# https://github.com/fastai/fastai/blob/master/courses/ml1/lesson4-mnist_sgd.ipynb



import os

import torch

import numpy as np

from fastai.vision import *   # Quick access to computer vision functionality
path = untar_data(URLs.MNIST_SAMPLE)

# path = untar_data(URLs.CIFAR)

data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)

data.normalize(imagenet_stats)

pass
data.show_batch()
# Essential components

# 1) Data (done)

# 2) Model (done)

# 3) Loss function (automatic)

# 4) Optimizer
model = models.resnet18()
# Define our learner

learn = Learner(data, model, metrics=accuracy, opt_func=torch.optim.Adam).to_fp16()
# Finding learning rate

learn.lr_find()

learn.recorder.plot()
# LR that we want is in range 1e-3~1e-2, lets use 1e-3

# Training

learn.fit(epochs=5, lr=1e-3)

learn.recorder.plot_losses()

learn.recorder.plot_metrics()
import torch.nn as nn

class flatten(nn.Module):

    def __init__(self):

        super().__init__()

        pass 

    

    def forward(self, x):

        return x.view(len(x), -1)
# Now let's build a custom cnn

custom_v1 = nn.Sequential(

    nn.Conv2d(3, 32, 3),

    nn.MaxPool2d(2),

    nn.ReLU(),

    

    nn.Conv2d(32, 32, 3),

    nn.MaxPool2d(2),

    nn.ReLU(),

    

    nn.Conv2d(32, 32, 3),

    nn.MaxPool2d(2),

    nn.ReLU(),

    

    flatten(),

    nn.Linear(1024, 10)

)
# We don't know whether 1024 is correct, so try it out!

dummy_input = torch.randn(32, 3, 28, 28)

custom_v1(dummy_input).shape
learn_v1 = Learner(data, custom_v1, metrics=accuracy, opt_func=torch.optim.Adam).to_fp16()

learn_v1.lr_find()

learn_v1.recorder.plot()
learn_v1.fit(epochs=5, lr=1e-3)

learn_v1.recorder.plot_losses()

learn_v1.recorder.plot_metrics()
### Not as good as before, eh? How can we make a better model?

# Now let's build a custom cnn

basic_block = lambda i, o: nn.Sequential(

                                nn.Conv2d(i, o, 3),

                                nn.MaxPool2d(2),

                                nn.BatchNorm2d(o),

                                nn.Dropout(.3),

                                nn.ReLU())



# What is BatchNorm?

# What is dropout? Why are they good?



# Note how I wrote this model. Why is this good?

custom_v2 = nn.Sequential(

    basic_block(3, 32),

    basic_block(32, 32),

    basic_block(32, 32),

    

    flatten(),

    nn.Linear(32, 10)

)
# We don't know whether 1024 is correct

dummy_input = torch.randn(32, 3, 28, 28)

custom_v2(dummy_input).shape
# Let's try this one out!

learn_v2 = Learner(data, custom_v2, metrics=accuracy, opt_func=torch.optim.Adam).to_fp16()

learn_v2.lr_find()

learn_v2.recorder.plot()
learn_v2.fit(epochs=5, lr=1e-3)

learn_v2.recorder.plot_losses()

learn_v2.recorder.plot_metrics()



# Why would this model perform better?