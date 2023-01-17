import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os





from fastai.vision import *

from fastai.metrics import error_rate

from fastai.vision import *

from torchvision.models import *

from glob import iglob

import cv2

import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

import os

from IPython.display import Image



%reload_ext autoreload

%autoreload 2

%matplotlib inline

data_dir='../input/stanford-car-dataset-by-classes-folder/car_data/car_data'

train_dir = "../input/stanford-car-dataset-by-classes-folder/car_data/car_data/train/*/*.jpg"

data_path = Path(data_dir)


fig, ax = plt.subplots(1,5, figsize=(20,4))

fig.suptitle('car image examples',fontsize=20)

# choose some images to plot

cnt = 1

plt.figure(1)



for img_path in iglob(train_dir):

    img = cv2.imread(img_path)

    plt.subplot(1,5,cnt)

    plt.imshow(img)

    #ax[0,cnt].imshow(img)

    cnt += 1

    if cnt > 5:

        break


# As we count the statistics, we can check if there are any completely black or white images

x_tot = np.zeros(3)

x2_tot = np.zeros(3)

cnt = 0



for img_path in iglob(train_dir):

    imagearray = cv2.imread(img_path).reshape(-1,3)/255.

    x_tot += imagearray.mean(axis=0)

    x2_tot += (imagearray**2).mean(axis=0)

    cnt += 1

    

channel_avr = x_tot/cnt

channel_std = np.sqrt(x2_tot/cnt - channel_avr**2)

channel_avr,channel_std
# Create ImageDataBunch using fastai data block API

batch_size = 64

data = ImageDataBunch.from_folder(data_path,  

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(do_flip=True,flip_vert=False, max_rotate=30, max_zoom=0.1, max_lighting=0.1),

                                  size=224,

                                  bs=batch_size, 

                                  num_workers=0).normalize([tensor([0.454952, 0.460148, 0.470733]), tensor([0.302969, 0.294664, 0.295581])])

                                  # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.
Image('../input/screen-shots/Screen Shot_model summary.png')
def getLearner():

    return cnn_learner(data, resnet34, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

learner = getLearner()
# some trick to make sure the pretrained weight gets downloaded correctly

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth
def getLearner():

    return cnn_learner(data, resnet34, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

learner = getLearner()
Image('../input/screen-shots/Screen Shot_resnet34_fit.png')
# We can use lr_find with different weight decays and record all losses so that we can plot them on the same graph

# Number of iterations is by default 100, but at this low number of itrations, there might be too much variance

# from random sampling that makes it difficult to compare WD's. I recommend using an iteration count of at least 300 for more consistent results.

lrs = []

losses = []

wds = [1e-6, 1e-5, 1e-4]

iter_count = 300



for wd in wds:

    learner = getLearner() #reset learner - this gets more consistent starting conditions

    learner.lr_find(wd=wd, num_it=iter_count)

    lrs.append(learner.recorder.lrs)

    losses.append(learner.recorder.losses)
_, ax = plt.subplots(1,1)

min_y = 4

max_y = 7

for i in range(len(losses)):

    ax.plot(lrs[i], losses[i])

    min_y = min(np.asarray(losses[i]).min(), min_y)

ax.set_ylabel("Loss")

ax.set_xlabel("Learning Rate")

ax.set_xscale('log')

#ax ranges may need some tuning with different model architectures 

ax.set_xlim((1e-4,3e-1))

ax.set_ylim((min_y - 0.02,max_y))

ax.legend(wds)

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
max_lr = 1e-2

wd = 1e-2

# 1cycle policy

learner_one_cycle = getLearner()

learner_one_cycle.fit_one_cycle(cyc_len=10, max_lr=max_lr, wd=wd)
learner_one_cycle.recorder.plot_lr()
# before we continue, lets save the model at this stage

learner_one_cycle.save('resnet34_stage1', return_path=True)
# unfreeze and run learning rate finder again

learner_one_cycle.unfreeze()

learner_one_cycle.lr_find(wd=wd)



# plot learning rate finder results

learner_one_cycle.recorder.plot()
# Now, smaller learning rates. This time we define the min and max lr of the cycle

learner_one_cycle.fit_one_cycle(cyc_len=10, max_lr=slice(5e-5,5e-4))

# Save the finetuned model

learner_one_cycle.save('resnet34_stage2')
# predict the validation set with our model

interp = ClassificationInterpretation.from_learner(learner_one_cycle)

interp.most_confused()
interp.plot_top_losses(9, figsize=(20,20))