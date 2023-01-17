!pip install -U -q sentencepiece ninja

!pip install -U -q git+https://github.com/fastai/fastai
import gc

from functools import partial

from pathlib import Path



from fastai.text import *

from fastai.callbacks import *

import numpy as np

import pandas as pd



home = Path(".")

!cp -R ../input/fastai-en-wiki-100kk-data-with-sentencepiece/* ./
bs= 512 + 128

max_vocab = 15_000
data_en_wiki = load_data(home, "data_en_wiki_15000", bs=bs)
data_en_wiki.show_batch()
1/0
learn.purge();

gc.collect()
config = awd_lstm_lm_config.copy()

wd=0.1

# config['qrnn'] = True

# config['n_hid'] = 1550 #default 1152

# config['n_layers'] = 4 #default 3

# wd=0.01
learn = language_model_learner(data_en_wiki, AWD_LSTM, config=config, drop_mult=0., true_wd=True, wd=wd,

                               pretrained=False, metrics=[accuracy, Perplexity()]).to_fp16()
# learn.lr_find()

# learn.recorder.plot(skip_end=10)
lr = 8e-04

lr *= bs/48  # Scale learning rate by batch size
learn.unfreeze()

learn.fit_one_cycle(10, lr, moms=(0.8,0.7),

                    callbacks=[

                        #SaveModelCallback(learn, monitor="perplexity", mode="min", name="best_model"),

                               ShowGraph(learn)]

                    )
learn.fit_one_cycle(5, lr/10, moms=(0.8,0.7))
learn.to_fp32().save(f"learn_en_wiki_{max_vocab}", with_opt=False)

learn.data.vocab.save(home/'models/learn_en_wiki_15_vocab.pkl',)
data_en_wiki = load_data(home, "data_en_wiki_15000_bwd", bs=bs, backwards=True)
learn = language_model_learner(data_en_wiki, AWD_LSTM, config=config, drop_mult=0., true_wd=True, wd=wd,

                               pretrained=False, metrics=[accuracy, Perplexity()]).to_fp16()
learn.unfreeze()

learn.fit_one_cycle(10, lr, moms=(0.8,0.7),

                    callbacks=[

                               #SaveModelCallback(learn, monitor="perplexity", mode="min", name="best_model"),

                               ShowGraph(learn)])
learn.fit_one_cycle(5, lr/10, moms=(0.8,0.7))
learn.to_fp32().save(f"learn_en_wiki_{max_vocab}_bwd", with_opt=False)

learn.data.vocab.save(home/'models/learn_en_wiki_15_vocab_bwd.pkl',)