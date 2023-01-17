from fastai.text import *

from fastai import *



import pandas as pd

import numpy as np
path = untar_data(URLs.IMDB)

path.ls()
!ls {path}/train/pos | head
from pytorch_pretrained_bert import BertTokenizer



bert_tok = BertTokenizer.from_pretrained(

    "bert-base-uncased",

)
class FastaiBertTokenizer(BaseTokenizer):

    '''wrapper for fastai tokenizer'''

    def __init__(self, tokenizer, max_seq=128, **kwargs):

        self._pretrained_tokenizer = tokenizer

        self.max_seq_length = max_seq

        

    def __call__(self,*args,**kwargs):

        return self

    

    def tokenizer(self,t):

        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_length - 2] + ['[SEP]']
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_bert_vocab.itos[2000:2010]
fastai_tokenizer = Tokenizer(tok_func=FastaiBertTokenizer(bert_tok,max_seq=128),pre_rules=[fix_html],post_rules=[])
processor = [OpenFileProcessor(),

             TokenizeProcessor(tokenizer=fastai_tokenizer,include_bos=False,include_eos=False),

             NumericalizeProcessor(vocab=fastai_bert_vocab)

            ]
data = (TextList

        .from_folder(path/'train',vocab=fastai_bert_vocab,processor=processor)

        .split_by_rand_pct(seed=42)

        .label_from_folder()

        .databunch(bs=16,num_workers=2)

       )
fastai_bert_vocab.stoi['the']
data.vocab.stoi['the']
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, BertForMaskedLM

bert_model_class = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model = bert_model_class
bert_1 = model.bert

embedder = bert_1.embeddings

pooler = bert_1.pooler

encoder = bert_1.encoder

classifier = [model.dropout,model.classifier]
n = len(encoder.layer) // 3

print(n)
model
from fastai.callbacks import *

learner = Learner(

 data, model,

 model_dir='/kaggle/working', metrics=accuracy

).to_fp16()
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(1,1e-4)
learner.fit(1,1e-4)
learner.recorder.plot_lr()