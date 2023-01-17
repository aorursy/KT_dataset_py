import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *

import warnings

warnings.filterwarnings("ignore")

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

config = Config(
    testing=False,
    #bert_model_name="bert-base-uncased",
    bert_model_name="bert-base-multilingual-uncased",
    
    max_lr=3e-5,
    epochs=4,
    use_fp16=True,
    bs=32,
    discriminative=False,
    max_seq_len=256,
)
from pytorch_pretrained_bert import BertTokenizer
bert_tok = BertTokenizer.from_pretrained(
    config.bert_model_name,
)
class FastAiBertTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

DATA_ROOT = Path("..")/"input"/ "jigsaw-multilingual-toxic-comment-classification/"

df1,df2,df3,test,sample = [pd.read_csv(DATA_ROOT / fname) for fname in ["jigsaw-toxic-comment-train.csv",
                                                                        "jigsaw-unintended-bias-train.csv",
                                                                        "validation.csv",
                                                                        "test.csv",
                                                                        "sample_submission.csv"
                                                                       ]]
df2.toxic = df2.toxic.round().astype(int)
train = pd.concat([
    df1[['comment_text', 'toxic']],
    df2[['comment_text', 'toxic']].query('toxic==1'),
    df2[['comment_text', 'toxic']].query('toxic==0').sample(n=200000, random_state=0)
])

# rankings_pd.rename(columns = {'test':'TEST', 'odi':'ODI', 
#                               't20':'T20'}, inplace = True) 
test.rename(columns={"content":"comment_text"}, inplace = True)

val = df3
test.head()
if config.testing:
    train = train.head(1024)
    val = val.head(1024)
    test = test.head(1024)

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, 
                                                          max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])
databunch = TextDataBunch.from_df(".", train, val, test,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_bert_vocab,
                  include_bos=False,
                  include_eos=False,
                  text_cols="comment_text",
                  label_cols="toxic",
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )
databunch.show_batch(10)
databunch.save(file = Path("data-jigsaw.pkl"))
test = load_data(path="/kaggle/working/", file = Path("data-jigsaw.pkl"))
test.show_batch()
os.chdir(r'/kaggle/working/')
from IPython.display import FileLink
FileLink(r'data-jigsaw.pkl')
