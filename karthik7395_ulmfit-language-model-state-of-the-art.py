# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!pip uninstall -y torch
!conda install -y pytorch=0.3.1.0 cuda80 -c soumith
import torch
torch.__version__
!pip install fastai==0.7.0
!pip uninstall -y torchtext
!pip install torchtext==0.2.3

from fastai.text import *
import html
train_df=pd.read_csv('../input/train.csv')
test_df=pd.read_csv('../input/test.csv')
trn_df,val_df = sklearn.model_selection.train_test_split(train_df, test_size=0.1)
trn_texts = trn_df['question_text']
val_texts = val_df['question_text']

trn_labels = trn_df['target']
val_labels = val_df['target']
trn_labels.value_counts()
col_names = ['labels','text']
BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)
!mkdir ../updated
!ls -lart ../
df_trn.to_csv('../updated/train.csv', header=False, index=False)
df_val.to_csv('../updated/test.csv', header=False, index=False)
chunksize=24000
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)
def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels
df_trn = pd.read_csv('../updated/train.csv', header=None, chunksize=chunksize)
df_val = pd.read_csv('../updated/test.csv', header=None, chunksize=chunksize)
tok_trn, trn_labels = get_all(df_trn, 1)
tok_val, val_labels = get_all(df_val, 1)
np.save('../updated/tok_trn.npy', tok_trn)
np.save('../updated/tok_val.npy', tok_val)
tok_trn = np.load('../updated/tok_trn.npy')
tok_val = np.load('../updated/tok_val.npy')
freq = Counter(p for o in tok_trn for p in o)
max_vocab = 60000
min_freq = 2
itos = [o for o,c in freq.most_common(max_vocab) if c>min_freq]
itos.insert(0, '_pad_')
itos.insert(0, '_unk_')
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
len(itos)
trn_lm = np.array([[stoi[o] for o in p] for p in tok_trn])
val_lm = np.array([[stoi[o] for o in p] for p in tok_val])
np.save('../updated/trn_ids.npy', trn_lm)
np.save('../updated/val_ids.npy', val_lm)
pickle.dump(itos, open('../updated/itos.pkl', 'wb'))
trn_lm = np.load('../updated/trn_ids.npy')
val_lm = np.load('../updated/val_ids.npy')
itos = pickle.load(open('../updated/itos.pkl', 'rb'))
vs=len(itos)
vs,len(trn_lm)
!wget -P ../updated/ http://files.fast.ai/models/wt103/fwd_wt103.h5 
wgts = torch.load('../updated/fwd_wt103.h5', map_location=lambda storage, loc: storage)
enc_wgts = to_np(wgts['0.encoder.weight'])
row_m = enc_wgts.mean(0)
PATH=Path('../updated')
!wget -P ../updated/ http://files.fast.ai/models/wt103/itos_wt103.pkl 
itos2 = pickle.load((PATH/'itos_wt103.pkl').open('rb'))
stoi2 = collections.defaultdict(lambda:-1, {v:k for k,v in enumerate(itos2)})
em_sz,nh,nl = 400,1150,3
new_w = np.zeros((vs, em_sz), dtype=np.float32)
for i,w in enumerate(itos):
    r = stoi2[w]
    new_w[i] = enc_wgts[r] if r>=0 else row_m
wgts['0.encoder.weight'] = T(new_w)
wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_w))
wgts['1.decoder.weight'] = T(np.copy(new_w))
import gc
gc.enable()
gc.collect()
wd=1e-7
bptt=70
bs=52
opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
trn_dl = LanguageModelLoader(np.concatenate(trn_lm), bs, bptt)
val_dl = LanguageModelLoader(np.concatenate(val_lm), bs, bptt)
md = LanguageModelData(PATH, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.7
learner= md.get_model(opt_fn, em_sz, nh, nl, 
    dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

learner.metrics = [accuracy]
learner.freeze_to(-1)
learner.model.load_state_dict(wgts)
lr=1e-2
lrs = lr
learner
### Taking too long to train as per the kernel requirement I couldn't commit it

#learner.fit(lrs, 1, wds=wd, use_clr=(32,2), cycle_len=1)
