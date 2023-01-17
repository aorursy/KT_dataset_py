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
from pytorch_pretrained_bert import BertTokenizer
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
    epochs=2,                   #4,
    use_fp16=True,
    bs=64,                      #32,
    discriminative=False,
    max_seq_len=192            #256
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
if config.testing:
    train = train.head(1024)
    val = val.head(1024)
    test = test.head(1024)

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
    df2[['comment_text', 'toxic']].query('toxic==0').sample(n=90000, random_state=0)
])

# rankings_pd.rename(columns = {'test':'TEST', 'odi':'ODI', 
#                               't20':'T20'}, inplace = True) 
test.rename(columns={"content":"comment_text"}, inplace = True)

val = df3

train.head()
test.head()
val.head()
bert_tok = BertTokenizer.from_pretrained(
    config.bert_model_name,
)
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
databunch.show_batch()
# databunch = load_data(path="../input/jigsawprocesseddatabunch/", file = Path("data-jigsaw.pkl"))
# databunch.show_batch()
# databunch.device
# VERSION = "20200325"  #@param ["1.5" , "20200325", "nightly"]
# !curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
# !python pytorch-xla-env-setup.py --version $VERSION
# import warnings
# import torch_xla
# import torch_xla.debug.metrics as met
# import torch_xla.distributed.data_parallel as dp
# import torch_xla.distributed.parallel_loader as pl
# import torch_xla.utils.utils as xu
# import torch_xla.core.xla_model as xm
# import torch_xla.distributed.xla_multiprocessing as xmp
# import torch_xla.test.test_utils as test_utils
# device = xm.xla_device()
# model = mx.to(device)
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=1)
class Loss_fn(nn.BCEWithLogitsLoss):
  __constants__ = ['weight', 'pos_weight', 'reduction']
  
  def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    
    super().__init__(size_average, reduce, reduction)
    self.register_buffer('weight', weight)
    self.register_buffer('pos_weight', pos_weight)

  def forward(self, input, target):
    # My target is of torch.Size([32])
    target = target.unsqueeze(1)   # Convert target size  of torch.Size([32, 1])
    target = target.float()        # BCE loss expects a Tensor of type float
  
    return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)
loss_func = Loss_fn()
from fastai.callbacks import *

learner = Learner(
    databunch, bert_model,
    loss_func=loss_func,
)
if config.use_fp16: learner = learner.to_fp16()

learner.model_dir = '/tmp/'
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(config.epochs, max_lr=config.max_lr)

def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
test_preds = get_preds_as_nparray(DatasetType.Test)

sample_submission = pd.read_csv("../input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
if config.testing: sample_submission = sample_submission.head(test.shape[0])
sample_submission['toxic'] = test_preds
sample_submission.to_csv("predictions.csv", index=False)

