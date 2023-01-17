!ls -l ../input/fastai-2012-wheel



!ls -l ../input/fast-v2-offline
!pip uninstall fastai transformers allennlp kornia tokenizers -y



!pip install ../input/fastai-2012-wheel/torch-1.6.0cu101-cp37-cp37m-linux_x86_64.whl



!pip install ../input/fastai-2012-wheel/tokenizers-0.8.1-cp37-cp37m-manylinux1_x86_64.whl



!pip install ../input/fastai-2012-wheel/kornia-0.4.0-py2.py3-none-any.whl



!pip install ../input/fastai-2012-wheel/fastcore-1.0.11-py3-none-any.whl



!pip install ../input/fast-v2-offline/dataclasses-0.6-py3-none-any.whl



!pip install ../input/fast-v2-offline/torchvision-0.7.0-cp37-cp37m-manylinux1_x86_64.whl



!pip install ../input/fastai-2012-wheel/fastai-2.0.12-py3-none-any.whl
import fastai

fastai.__version__
import torch

torch.cuda.is_available()
%pylab inline
from fastai.tabular.all import *



import pandas as pd

import numpy as np

import os, re



from os.path import isdir, isfile, basename, dirname



from glob import glob

from pathlib import Path as P

from tqdm.notebook import tqdm



import seaborn as sns



from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss





def random_seed(seed_value, use_cuda=True):

    # https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/5

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    random.seed(seed_value) # Python

    if use_cuda: 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False



plt.rcParams['figure.facecolor'] = 'w'

plt.rcParams['figure.dpi'] = 150

pd.options.display.max_colwidth = 100

pd.options.display.max_columns = 100





PATH = '../input/lish-moa'

print(glob(f'{PATH}/*'))



scored = pd.read_csv(f'{PATH}/train_targets_scored.csv')

train_feat = pd.read_csv(f'{PATH}/train_features.csv')

test = pd.read_csv(f'{PATH}/test_features.csv').set_index('sig_id')



train = pd.merge(train_feat, scored, on='sig_id')



cont_names = train_feat.filter(regex='g-|c-').columns.to_list()

y_names_scored = scored.set_index('sig_id').columns.to_list()

y_names = y_names_scored



random_seed(42)

def predict(learn, data):

    dl = learn.dls.test_dl(data)

    pred, _ = learn.get_preds(dl=dl)        

    pred_scored = pred[:,:len(y_names_scored)].numpy()

    return pred_scored



def create_submission(learn=None, test=None, from_numpy=False, fn=None):

    if fn is None:

        fn = 'submission.csv'

    if from_numpy is False:       

        pred_scored = predict(learn, test)

    else:

        pred_scored = from_numpy

    submission = pd.read_csv(f'{PATH}/sample_submission.csv').set_index('sig_id')

    submission.loc[:, y_names_scored] = pred_scored

    submission.loc[test['cp_type']=='ctl_vehicle', y_names_scored] = 0

    submission.to_csv(fn)

    return pred_scored
config = tabular_config(ps=[0.8, 0.6] , embed_p=0.5)

layers = [2018, 1024]

batch_size = 128



splits = RandomSplitter(valid_pct=0.1)(range_of(train))



to = TabularPandas(train, y_names=y_names,

     cat_names = ['cp_type', 'cp_time', 'cp_dose'],

     cont_names = cont_names,

     procs = [Categorify, FillMissing, Normalize],

     splits = splits)



dls = to.dataloaders(bs=batch_size)



callbacks = [

    EarlyStoppingCallback(monitor='valid_loss', min_delta=1e-5, patience=10),    

    ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=5, min_lr=1e-8),

    SaveModelCallback(name='model'),

] 



def fitting_procedure(learn):

    learn.lr_find()

    show()

    learn.fit_one_cycle(10, 2e-2, wd=0.1)

    learn.fit_one_cycle(10, slice(1e-3), wd=0.2)

    learn.recorder.plot_loss(with_valid=True)

    show()

    learn.fit(100, slice(1e-3), wd=0.5, cbs=callbacks)

    learn.recorder.plot_loss(with_valid=True)

    show()

    learn.load('model')

    return learn
preds = []



for i in tqdm( range(5) ):

    print('='*30+f' Round {i} '+'='*30)

    learn = tabular_learner(dls, layers=layers, 

                            config=config, opt_func=Adam,

                            loss_func=BCEWithLogitsLossFlat())

    print(learn.model)

    learn = fitting_procedure(learn)



    val_score = learn.validate()[0]

    

    print(f'Best validation score: {val_score}.')

    min_val_score = val_score

    preds.append( predict(learn,test) )
print('Done')
pred = np.array(preds).mean(axis=0)



create_submission(from_numpy=pred, test=test)
! ls
pred.shape