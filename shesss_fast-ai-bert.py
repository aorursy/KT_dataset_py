import numpy as np

import pandas as pd

from os.path import join



from typing import *



import torch

import torch.optim as optim



from pytorch_pretrained_bert import BertTokenizer
from fastai import *

from fastai.vision import *

from fastai.text import *

from fastai.callbacks import *
import math

import torch

from torch.optim.optimizer import Optimizer, required



class RAdam(Optimizer):



    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        self.buffer = [[None, None, None] for ind in range(10)]

        super(RAdam, self).__init__(params, defaults)



    def __setstate__(self, state):

        super(RAdam, self).__setstate__(state)



    def step(self, closure=None):



        loss = None

        if closure is not None:

            loss = closure()



        for group in self.param_groups:



            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data.float()

                if grad.is_sparse:

                    raise RuntimeError('RAdam does not support sparse gradients')



                p_data_fp32 = p.data.float()



                state = self.state[p]



                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p_data_fp32)

                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                else:

                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)

                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)



                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']



                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)



                state['step'] += 1

                buffered = self.buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:

                    N_sma, step_size = buffered[1], buffered[2]

                else:

                    buffered[0] = state['step']

                    beta2_t = beta2 ** state['step']

                    N_sma_max = 2 / (1 - beta2) - 1

                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                    buffered[1] = N_sma



                    # more conservative since it's an approximated value

                    if N_sma >= 5:

                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])

                    else:

                        step_size = group['lr'] / (1 - beta1 ** state['step'])

                    buffered[2] = step_size



                if group['weight_decay'] != 0:

                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)



                # more conservative since it's an approximated value

                if N_sma >= 5:            

                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)

                else:

                    p_data_fp32.add_(-step_size, exp_avg)



                p.data.copy_(p_data_fp32)



        return loss
config = {

    'testing':False,

    'bert_model_name':"bert-base-uncased",

    'max_lr':3e-5,

    'epochs':6,

    'use_fp16':True,

    'bs':32,

    'discriminative':False,

    'max_seq_len':256,

}

data_path = '../input'
bert_tok = BertTokenizer.from_pretrained(config['bert_model_name'])
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
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config['max_seq_len']), pre_rules=[], post_rules=[])

fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
train_df = pd.read_csv(join(data_path, 'train.csv'))

val_df = pd.read_csv(join(data_path, 'valid.csv'))

test_df = pd.read_csv(join(data_path, 'test.csv'))



if config['testing']:

    train_df = train_df.head(1024)

    val_df = val_df.head(1024)

    test_df = test_df.head(1024)
train_df.head()
print(sorted(train_df['label'].unique()))

print(sorted(val_df['label'].unique()))
label_cols = list(pd.get_dummies(train_df['label']).columns)

print(label_cols)
databunch = TextDataBunch.from_df(".", train_df, val_df, test_df,

                  tokenizer=fastai_tokenizer,

                  vocab=fastai_bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols="text",

                  label_cols='label',

                  bs=config['bs'],

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
# databunch.save('databunch')
# databunch = load_data('.', 'databunch');
databunch.show_batch()
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

bert_model = BertForSequenceClassification.from_pretrained(config['bert_model_name'], num_labels=len(label_cols))
loss_func = nn.CrossEntropyLoss()
from fastai.callbacks import *



learner = Learner(

    databunch, bert_model,

    loss_func=loss_func,

    opt_func=RAdam,

)

if config['use_fp16']: 

    learner = learner.to_fp16()
learner.lr_find()
learner.recorder.plot()
learner.freeze_to(-1)
learner.fit_one_cycle(1, max_lr=config['max_lr'])
learner.unfreeze()
learner.fit_one_cycle(config['epochs'], max_lr=config['max_lr'], 

                      callbacks=[SaveModelCallback(learner, name='best',

                                                 every='improvement', monitor='valid_loss')])
learner.load('best');
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

test_preds
test_preds = np.argmax(test_preds, axis=1)

test_preds
label_dict = {i: label for i, label in enumerate(learner.data.classes)}

label_dict
sub_df = pd.read_csv(join(data_path, 'sample_submission.csv'), index_col='id')

sub_df['label'] = test_preds

sub_df['label'] = sub_df['label'].apply(lambda x: label_dict[x])
sub_df.head()
sub_df.to_csv('submission.csv')