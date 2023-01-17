learners_names = ['fit', 'fit_one_cycle()', 'fit_one_cycle(5, max_lr)', 'fit_one_cycle(5, max_lr=[])', 'custom cosine annealing', 'custom linear annealing', 'custom NO annealing' ]
import numpy as np

from copy import copy

import torch

import cv2

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import PIL

print(PIL.PILLOW_VERSION)



np.random.seed(2)



from fastai.vision import *

from fastai.metrics import error_rate

import fastai

fastai.__version__
import os

import pandas as pd
train_folder = '../input/images/images'

print(os.listdir(train_folder)[:10])
fnames = get_image_files(train_folder)

print(fnames[:5])

pat = re.compile(r'/([^/]+)_\d+.jpg$') # we specify a regex for finding cat or dog images

sz = 64

bs = 64

data = ImageDataBunch.from_name_re(

                                train_folder,

                                fnames,

                                pat,

                                ds_tfms=get_transforms(),

                                size=sz, bs=bs,

                                valid_pct = 0.75, #_______________________________________________________

                                num_workers = 0, # for code safety on kaggle

).normalize(imagenet_stats)

data
print(data.classes)

data.show_batch()
learn = cnn_learner(

    data,

    models.resnet18,

    metrics=error_rate,

    model_dir="/tmp/model/"

)
learn.unfreeze()
learn1 = copy(learn)

learn2 = copy(learn)

learn3 = copy(learn)

learn4 = copy(learn)

learn5 = copy(learn)

learn6 = copy(learn)

learn7 = copy(learn)


def learner_plots(learner, name):

    pltName = name

    metricsName =  'error_rate'

    lrec = learner.recorder

    plt.plot(figsize=(2,9))



    plt.subplot(4, 1, 1)

    plt.plot( lrec.lrs, lrec.losses )

    plt.title(pltName, fontdict={'fontsize':20})

    plt.ylabel('Losses', fontdict={'fontsize':15})

    plt.xlabel('LR')



    plt.subplot(4, 1, 2)

    plt.plot( lrec.lrs )

    plt.ylabel('LR', fontdict={'fontsize':15})

    plt.xlabel('Iterations')



    plt.subplot(4, 1, 3)

    iters = np.arange(len(lrec.lrs))

    val_iter = np.cumsum(lrec.nb_batches)

    plt.plot( iters, lrec.losses )

    plt.plot( val_iter, lrec.val_losses )

    plt.ylabel('Loss')

    plt.yscale('log')

    plt.xlabel('Iterations')



    plt.subplot(4, 1, 4)

    # plt.bar( 

    #     np.arange(len(learn1.recorder.metrics))+1,

    #     np.array(learn1.recorder.metrics[0]).astype(int)

    # )

    plt.plot(lrec.metrics)

    plt.ylabel(metricsName)

    plt.xlabel('Iterations 1e-2')

    plt.gcf().set_figheight(15)

    plt.gcf().set_figwidth(15)

    return
def plotLearners(learners, learners_names, figsize=(20, 6)):

    '''

    edit: https://github.com/fastai/fastai/blob/master/fastai/basic_train.py#L524

    '''

    # figure(figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    nl = len(learners)

    fig, ax = plt.subplots(3,nl)

    fig.set_figheight(figsize[0])

    fig.set_figwidth(figsize[1])

    for li in range(len(learners)):

        lrec = learners[li].recorder

        skip_start = 10

        skip_end = 5

        lrs = lrec._split_list(lrec.lrs, skip_start, skip_end)

        losses = [x.item() for x in lrec._split_list(lrec.losses, skip_start, skip_end)]

        val_losses =  learners[li].recorder.val_losses

        # if 'k' in kwargs: losses = self.smoothen_by_spline(lrs, losses, **kwargs)

        # plt.subplot(2,4,li);

        # PLOT LR vs LOSS

        ax[0, li].plot(lrs, losses);

        if (li==0): ax[0, li+1].set_ylabel("Loss")

        ax[0, li].set_title(learners_names[li])

        ax[0, li].set_xlabel("Learning Rate")

        ax[0, li].set_xscale('log')

        show_moms=False; skip_start=0; skip_end=0; return_fig=None;

        # PLOT LR vs ITERs

        lrs = lrec._split_list(lrec.lrs, skip_start, skip_end)

        iterations = lrec._split_list(range_of(lrec.lrs), skip_start, skip_end)

        if show_moms: # not used right here

            moms = lrec._split_list(self.moms, skip_start, skip_end)

            fig, axs = plt.subplots(1,2, figsize=(12,4))

            axs[0].plot(iterations, lrs)

            axs[0].set_xlabel('Iterations')

            axs[0].set_ylabel('Learning Rate')

            axs[1].plot(iterations, moms)

            axs[1].set_xlabel('Iterations')

            axs[1].set_ylabel('Momentum')

        else:

            # plt.subplot(2,4, (li+1)+4)

            ax[1, li].plot(iterations, lrs)

            ax[1, li].set_xlabel('Iterations')

            ax[1, li].set_ylabel('Learning Rate')

        # PLOT LOSSES

        losses = lrec._split_list(lrec.losses, skip_start, skip_end)

        ax[2, li].plot(iterations, losses, label='Train') ## SUBPLOT 3

        val_iter = lrec._split_list_val(np.cumsum(lrec.nb_batches), skip_start, skip_end)

        val_losses = lrec._split_list_val(lrec.val_losses, skip_start, skip_end)

        ax[2, li].plot(val_iter, val_losses, label='Validation')

        ax[2, li].set_ylabel('Loss')

        ax[2, li].set_xlabel('Batches processed')

        ax[2, li].legend()

        if ifnone(return_fig, defaults.return_fig): return fig

        # PLOT METRICS

#         val_iter = self._split_list_val(np.cumsum(self.nb_batches), skip_start, skip_end)

#         axes = axes.flatten() if len(self.metrics[0]) != 1 else [axes]

#         for i, ax in enumerate(axes):

#             values = [met[i] for met in self.metrics]

#             values = self._split_list_val(values, skip_start, skip_end)

#             ax.plot(val_iter, values)

#             ax.set_ylabel(str(self.metrics_names[i]))

#             ax.set_xlabel('Batches processed') 

#     plt.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
%%time

learn1.lr_find()

learn2.lr_find()

learn3.lr_find()

learn4.lr_find()

# learn5.lr_find()

# learn6.lr_find()

# learn7.lr_find()
learners = [learn1.recorder, learn2.recorder, learn3.recorder, learn4.recorder]

plotLearners(learners, learners_names, figsize=(15,18))
learn3.recorder.plot(suggestion=True)

mng3 = learn3.recorder.min_grad_lr  #// Suggested-LR, https://docs.fast.ai/callbacks.lr_finder.html
learn1 = copy(learn)

learn2 = copy(learn)

learn3 = copy(learn)

learn4 = copy(learn)
learn1.fit(5)     # 10epochs, no cyclical_lr
# plotLearners([learn1], learners_names[:1], figsize=(20, 6))

learner_plots(learn1, 'fit(5)')
learn2.fit_one_cycle(5)         # cyclical_lr
learner_plots(learn2, 'fit_one_cycle(5)')
learn3.lr_find()

learn3.recorder.plot(suggestion=True)

mng3 = learn3.recorder.min_grad_lr  #// Suggested-LR, https://docs.fast.ai/callbacks.lr_finder.html
learn3.fit_one_cycle(5, max_lr=mng3); print('Training learn3 with min numerical gradient: ', mng3)  # cyclical_lr with max value
learner_plots(learn3, 'fit_one_cycle + max_lr')
learn4.fit_one_cycle(5, max_lr=slice(mng3*0.01, mng3))    # cyclical_lr + differential_lr
learner_plots(learn4, 'fit_one_cycle + differential LR')
from fastai.callbacks.general_sched import *
def fit_custom_annealing(learn, anneal, n_cycles, lr, mom, cycle_len, cycle_mult):

    n = len(learn.data.train_dl)

    phases = [(TrainingPhase(n * (cycle_len * cycle_mult**i))

                 .schedule_hp('lr', lr, anneal=anneal)

                 .schedule_hp('mom', mom)) for i in range(n_cycles)]

    sched = GeneralScheduler(learn, phases)

    learn.callbacks.append(sched)

    if cycle_mult != 1:

        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles)/(1-cycle_mult)) 

    else: total_epochs = n_cycles * cycle_len

    learn.fit(total_epochs)
anneal = annealing_cos

fit_custom_annealing(learn5, anneal=anneal, n_cycles=3, lr=1e-3, mom=0.9, cycle_len=1, cycle_mult=2)
learner_plots(learn5, 'Cosine annealing')
anneal = annealing_linear

fit_custom_annealing(learn6, anneal=anneal, n_cycles=3, lr=1e-3, mom=0.9, cycle_len=1, cycle_mult=2)
learner_plots(learn6, 'Linear annealing')
anneal = annealing_no

fit_custom_annealing(learn7, anneal=anneal, n_cycles=3, lr=1e-3, mom=0.9, cycle_len=1, cycle_mult=2)
learner_plots(learn7, 'No annealing')
learners = [learn1, learn2, learn3, learn4]

plotLearners(learners, learners_names,  figsize=(15,18))
learners = [learn5, learn6, learn7]

plotLearners(learners, learners_names,  figsize=(15,18))