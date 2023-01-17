!python -m fastai.utils.show_install
%reload_ext autoreload

%autoreload 2

%matplotlib inline



# For language modelling

from fastai import *

from fastai.text import *

from fastai.callbacks import *

from fastai.metrics import *



from pathlib import Path



DATA_PATH = Path('/kaggle/input/data-bunch-object-creation')

OUT_PATH = Path('/kaggle/working')



#hide

seed = 0



# python RNG

import random

random.seed(seed)



# pytorch RNGs

import torch

torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)



# numpy RNG

import numpy as np

np.random.seed(seed)
data_lm = load_data(DATA_PATH, 'data_lm.pkl', bs=64)
# data_lm.save(DATA_PATH/'data_lm.pkl')

data_lm.show_batch()
# To use qrnn

config = awd_lstm_lm_config.copy()

config['qrnn'] = True



perplexity = Perplexity()



learn = language_model_learner(data_lm, arch=AWD_LSTM, config=config,

                               drop_mult=0.3,

                               pretrained=False,

                                metrics=[accuracy, perplexity],

                              ).to_fp16()



learn.model_dir=OUT_PATH/'model_dir'
learn.lr_find()
learn.recorder.plot(suggestion=True)
min_grad_lr = learn.recorder.min_grad_lr

min_grad_lr
cb_list = [SaveModelCallback(learn, every='improvement', mode='min', monitor='perplexity', name='best_st1'),

           ReduceLROnPlateauCallback(learn, monitor='perplexity', mode='min', patience=3, min_delta=2, min_lr=1e-5),

           CSVLogger(learn, filename=OUT_PATH/'learner_history', append=True)]
learn.fit_one_cycle(50, min_grad_lr,

                    # Momentums, just a try!

                    div_factor=100.0, pct_start=0.8, moms=(0.75,0.65),

                    callbacks=cb_list)
learn.load('best_st1');

learn.save('ta-wiki-stage1')

learn.save_encoder('ta-wiki-enc-stage1')

learn.export(learn.model_dir/'ta-lang_mod_st1.pkl')
learn.csv_logger.read_logged_file()