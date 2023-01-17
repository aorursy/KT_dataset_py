!pip install transformers -q
import numpy as np

import pandas as pd

from pathlib import Path

import os

import torch

import torch.optim as optim

import random



import fastai

from fastai import *

from fastai.text import *

from fastai.callbacks import *



import transformers

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
def seed_all(seed_value):

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False

        

seed_all(42)
PATH = Path('../input/jigsaw-multilingual-toxic-comment-classification')

df_train = pd.read_csv(PATH/'jigsaw-toxic-comment-train.csv')

df_train = df_train.drop(['severe_toxic','obscene','threat','insult','identity_hate'], axis=1)

df_train['valid'] = False



df_valid = pd.read_csv(PATH/'validation.csv')

df_valid = df_valid.drop(['lang'], axis=1)

df_valid['valid'] = True



df_test =  pd.read_csv(PATH/'test.csv')

df_train_all = pd.concat([df_train, df_valid], axis=0)
model_class     = BertForSequenceClassification

tokenizer_class = BertTokenizer

config_class    = BertConfig
class TransformersBaseTokenizer(BaseTokenizer):

    """

    Wrapper around PreTrainedTokenizer to be compatible with fastai.

    """

    

    def __init__(self, pretrained_tokenizer, **kwargs):

        self._pretrained_tokenizer = pretrained_tokenizer

        self.max_seq_len = pretrained_tokenizer.max_len

        

    def __call__(self, *args, **kwargs):

        return self

    

    def tokenizer(self, t:str) -> List[str]:  # replace the `tokenizer` function in the parent class

        """

        Limits the maximum sequence length and add the special tokens

        """

        # Get the CLS and SEP tokens

        CLS = self._pretrained_tokenizer.cls_token  # method found in `pretrained_tokenizer`

        SEP = self._pretrained_tokenizer.sep_token

        

        # tokenize the string `t` pass into this function

        tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

        

        return [CLS] + tokens + [SEP]

    

transformer_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer=transformer_tokenizer)

fastai_tokenizer = Tokenizer(tok_func=transformer_base_tokenizer, pre_rules=[], post_rules=[])
class TransformersVocab(Vocab):

    def __init__(self, tokenizer: PreTrainedTokenizer):

        super().__init__(itos=[])

        self.tokenizer = tokenizer

        

    def numericalize(self, t:Collection[str]) -> List[int]:

        "Convert a list of tokens `t` into their ids."

        return self.tokenizer.convert_tokens_to_ids(t)

    

    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums) 

    

    def __getstate__(self):

        return {'itos':self.itos, 'tokenizer':self.tokenizer}



    def __setstate__(self, state:dict):

        self.itos = state['itos']

        self.tokenizer = state['tokenizer']

        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
transformer_vocab      = TransformersVocab(tokenizer=transformer_tokenizer)

numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)   # fastai

tokenize_processor     = TokenizeProcessor(tokenizer=fastai_tokenizer,

                                           include_bos=False,

                                           include_eos=False)

transformers_processor = [tokenize_processor, numericalize_processor]
databunch = load_data(path='', file='../input/fastai-dataloaders/jigsaw_data.pkl', 

                      bs=32, pad_first=False, pad_idx=transformer_tokenizer.pad_token_id)
# databunch = (TextList.from_df(df_train_all, cols='comment_text', processor=transformers_processor)

#              .split_from_df(col='valid')

#              .label_from_df(cols=['toxic'])

#              .add_test(df_test)

#              .databunch(bs=32, pad_first=False, pad_idx=transformer_tokenizer.pad_token_id))
class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):

        super().__init__()

        self.transformer = transformer_model

        

    def forward(self, input_ids):

        logits = self.transformer(input_ids)[0] # first element of the output tuple: `last_hidden_state` - from the docs

        return logits
config = BertConfig.from_pretrained('bert-base-multilingual-uncased')

config.num_labels = 2

transformer_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', config=config)

custom_transformer_model = CustomTransformerModel(transformer_model)
from transformers import AdamW

learner = Learner(    data = databunch, 

                     model = custom_transformer_model, 

                  opt_func = lambda input:AdamW(input, correct_bias=False),

                   metrics = AUROC())

learner.model_dir = '../input/models/'

learner.load('bert_fastai_unfrozen_last_12')
def get_preds_as_nparray(ds_type) -> np.ndarray:

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]



test_preds = get_preds_as_nparray(DatasetType.Test)
sample_submission = pd.read_csv(PATH/'sample_submission.csv')

sample_submission['toxic'] = test_preds[:,1] 

sample_submission.to_csv('submission.csv', index=False)