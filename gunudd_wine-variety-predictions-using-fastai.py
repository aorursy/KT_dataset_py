import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai.text import *
from fastai import *
import re
df = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',index_col=0)
df.shape
df.head()
data_lm = (TextList.from_df(df=df,path='.',cols='description') 
            .random_split_by_pct(0.1)
            .label_for_lm()           
            .databunch(bs=48))
data_lm.save('tmp_lm')
data_lm.vocab.itos[:10]
data_lm.train_ds[0][0]
data_lm.train_ds[0][0].data[:10]
data_lm = TextLMDataBunch.load('.', 'tmp_lm', bs=48)
data_lm.show_batch()
fnames=['../input/wt1031/itos_wt103.pkl',
       '../input/wt1031/lstm_wt103.pth']
def language_model_learner(data:DataBunch, bptt:int=70, emb_sz:int=400, nh:int=1150, nl:int=3, pad_token:int=1,
                  drop_mult:float=1., tie_weights:bool=True, bias:bool=True, qrnn:bool=False, pretrained_model=None,
                  pretrained_fnames:OptStrTuple=None, **kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data`."
    dps = default_dropout['language'] * drop_mult
    vocab_size = len(data.vocab.itos)
    model = get_language_model(vocab_size, emb_sz, nh, nl, pad_token, input_p=dps[0], output_p=dps[1],
                weight_p=dps[2], embed_p=dps[3], hidden_p=dps[4], tie_weights=tie_weights, bias=bias, qrnn=qrnn)
    learn = LanguageLearner(data, model, bptt, split_func=lm_split, **kwargs)
    if pretrained_model is not None:
        model_path = Path('../input/wt1031/')
        fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        learn.load_pretrained(*fnames)
        learn.freeze()
    if pretrained_fnames is not None:
        fnames = [learn.path/learn.model_dir/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        learn.load_pretrained(*fnames)
        learn.freeze()
    return learn
learn = language_model_learner(data_lm,path='.', pretrained_model=' ', drop_mult=0.3)
learn.lr_find()
learn.recorder.plot(skip_end=10)
learn.fit_one_cycle(1, 5e-2)
learn.save('fit_head')
learn.load('fit_head');
learn.unfreeze()
learn.fit_one_cycle(5, 1e-3)
learn.save('fine_tuned')
learn.load('fine_tuned');
TEXT = "i taste hints of"
N_WORDS = 40
N_SENTENCES = 2
print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
learn.save_encoder('fine_tuned_enc')
min_samples=10
lst=df.variety.value_counts()
wines=lst[lst>min_samples].keys()
subdf=df[df.variety.isin(wines)]
subdf.shape,df.shape
data_clas = (TextList.from_df(df=subdf,path='.',cols='description', vocab=data_lm.vocab)
             .random_split_by_pct(0.1)
             .label_from_df('variety')
             .databunch(bs=48))
data_clas.save('tmp_clas')
data_clas = TextClasDataBunch.load('.', 'tmp_clas', bs=48)
data_clas.show_batch()
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')
learn.freeze()
learn.lr_find()
learn.recorder.plot(skip_end=8)
learn.fit_one_cycle(1, 2e-2)
print ('')
learn.save('first')
learn.load('first');
for i in range(2,5):
    learn.freeze_to(-i)
    learn.fit_one_cycle(1,slice((1*10**-i)/(2.6**4),1*10**-i))
    learn.save('sub-'+str(i))
    print ('')
learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-5/(2.6**4),1e-3))
print ('')
learn.save('final')
learn.fit_one_cycle(5, slice(1e-5,1e-3))
print ('')
learn.save('final')
learn.show_results(rows=10)
learn.predict("tannins are well proportioned both grained and supple")[0]
learn.predict("a light wine with hints of bitterness and fruit")[0]
learn.predict("a wine full of flavor and color, mostly white")[0]