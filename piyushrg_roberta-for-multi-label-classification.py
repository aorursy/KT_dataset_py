import numpy as np
import pandas as pd
path = '../input/janatahack-independence-day-2020-ml-hackathon/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
submit = pd.read_csv(path+'sample_submission_UVKGLZE.csv')
test.head()
train.head()
print(train.shape, test.shape, submit.shape)
print(len(np.intersect1d(train.TITLE, test.TITLE)))
print(len(np.intersect1d(train.ID, test.ID)))
cols = ['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']
from fastai import *
from fastai.text import *
import os
# !pip install -q transformers
from pathlib import Path 

import os

import torch
import torch.optim as optim

import random 

from fastai.callbacks import *

# transformers
from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import AlbertForSequenceClassification, AlbertTokenizer, AlbertConfig
import fastai
import transformers
print('fastai version :', fastai.__version__)
print('transformers version :', transformers.__version__)
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig),
    'albert':(AlbertForSequenceClassification,AlbertTokenizer, AlbertConfig)
}
# Parameters
seed = 10
use_fp16 = True
bs = 4

model_type = 'roberta'
pretrained_model_name = 'roberta-large'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
def seed_all(seed_value):
    random.seed(seed_value) # Python
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
seed_all(seed)
class TransformersBaseTokenizer(BaseTokenizer):
    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""
    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):
        self._pretrained_tokenizer = pretrained_tokenizer
        self.max_seq_len = pretrained_tokenizer.max_len
        self.model_type = model_type

    def __call__(self, *args, **kwargs): 
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length and add the spesial tokens"""
        CLS = self._pretrained_tokenizer.cls_token
        SEP = self._pretrained_tokenizer.sep_token
        if self.model_type in ['roberta']:
            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]
            tokens = [CLS] + tokens + [SEP]
        else:
            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]
            if self.model_type in ['xlnet']:
                tokens = tokens + [SEP] +  [CLS]
            else:
                tokens = [CLS] + tokens + [SEP]
        return tokens
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
class TransformersVocab(Vocab):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super(TransformersVocab, self).__init__(itos = [])
        self.tokenizer = tokenizer
    
    def numericalize(self, t:Collection[str]) -> List[int]:
        "Convert a list of tokens `t` to their ids."
        return self.tokenizer.convert_tokens_to_ids(t)
        #return self.tokenizer.encode(t)

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
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)

tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)

transformer_processor = [tokenize_processor, numericalize_processor]
# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        
        # attention_mask
        # Mask to avoid performing attention on padding token indices.
        # Mask values selected in ``[0, 1]``:
        # ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 
        
        logits = self.transformer(input_ids,
                                  attention_mask = attention_mask)[0]   
        return logits
config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 2
config.use_bfloat16 = use_fp16
#print(config)
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id
#col = 'Computer Science'
def prep_data_bunch(col):
    data_transclas = (TextList.from_df(train, cols=['ABSTRACT','TITLE'], processor=transformer_processor)
                      .split_by_rand_pct(0.1,seed=seed)
                      .label_from_df(cols= col)
                      .add_test(test)
                      .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))
    
    return data_transclas
#data_transclas = prep_data_bunch(col)
#print('[CLS] token :', transformer_tokenizer.cls_token)
#print('[SEP] token :', transformer_tokenizer.sep_token)
#print('[PAD] token :', transformer_tokenizer.pad_token)
#data_transclas.show_batch()
#print('[CLS] id :', transformer_tokenizer.cls_token_id)
#print('[SEP] id :', transformer_tokenizer.sep_token_id)
#print('[PAD] id :', pad_idx)
#test_one_batch = data_transclas.one_batch()[0]
#print('Batch shape : ',test_one_batch.shape)
#print(test_one_batch)
from fastai.callbacks import *
from transformers import AdamW
from functools import partial

CustomAdamW = partial(AdamW, correct_bias=False)
def create_learner_instance(data_transclas):
    learner = Learner(data_transclas, 
                      custom_transformer_model, 
                      opt_func = CustomAdamW, 
                      metrics=[accuracy, error_rate])

    # Show graph of learner stats and metrics after each epoch.
    learner.callbacks.append(ShowGraph(learner))

    # Put learn in FP16 precision mode. --> Seems to not working
    if use_fp16: learner = learner.to_fp16()
    
    list_layers = [learner.model.transformer.roberta.embeddings,
              learner.model.transformer.roberta.encoder.layer[0],
              learner.model.transformer.roberta.encoder.layer[1],
              learner.model.transformer.roberta.encoder.layer[2],
              learner.model.transformer.roberta.encoder.layer[3],
              learner.model.transformer.roberta.encoder.layer[4],
              learner.model.transformer.roberta.encoder.layer[5],
              learner.model.transformer.roberta.encoder.layer[6],
              learner.model.transformer.roberta.encoder.layer[7],
              learner.model.transformer.roberta.encoder.layer[8],
              learner.model.transformer.roberta.encoder.layer[9],
              learner.model.transformer.roberta.encoder.layer[10],
              learner.model.transformer.roberta.encoder.layer[11],
              learner.model.transformer.roberta.pooler]
    
    learner.split(list_layers)
    num_groups = len(learner.layer_groups)
    #print('Learner split in',num_groups,'groups')
    #print(learner.layer_groups)
    #print(learner.model)
    return learner, num_groups

#learner = create_learner_instance(data_transclas)
def train_model(learner, num_groups, lr, epochs, unfreeze_all):
    
    if unfreeze_all == True:
        learner.unfreeze()
        learner.fit_one_cycle(epochs, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
    else:
        learner.freeze_to(-2)
        learner.fit_one_cycle(epochs,max_lr=slice(lr*0.9**num_groups, lr),moms=(0.8,0.9))
    
    return learner
#learner = train_model(learner)
def get_preds_as_nparray(ds_type, learner, data_transclas):
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in data_transclas.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
def create_sub_file(col, learner, data_transclas):
    print('Predicting..')
    test_preds = get_preds_as_nparray(DatasetType.Test, learner, data_transclas)
    print('Predictions done!')
    submit[col] = np.argmax(test_preds,axis=1)
    return 
def main(col, lr, epochs, unfreeze_all):
    print('Creating DataBunch!')
    data_transclas = prep_data_bunch(col)
    print('Prepared DataBunch!')
    
    print('Creating Learner instance!')
    learner, num_groups = create_learner_instance(data_transclas)
    print('Prepared Learner instance!')
    
    print('Training Model!')
    learner = train_model(learner, num_groups, lr, epochs, unfreeze_all)
    print('Trained Complete!')
    
    print('Creating Submission File!')
    create_sub_file(col, learner, data_transclas)
    print('Submission file created!')
    
    return
%%time
cols = ['Computer Science', 'Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']

lr_dict = {'Computer Science':1e-5
          , 'Physics': 1e-5
          , 'Mathematics': 1e-5
          , 'Statistics': 1e-5
          , 'Quantitative Biology': 1e-4
          , 'Quantitative Finance': 1e-4}
epoch_dict = {'Computer Science':1
          , 'Physics': 1
          , 'Mathematics': 1
          , 'Statistics': 1
          , 'Quantitative Biology': 1
          , 'Quantitative Finance': 1}

tuning = {'Computer Science':False
          , 'Physics': False
          , 'Mathematics': False
          , 'Statistics': False
          , 'Quantitative Biology': False
          , 'Quantitative Finance': False}

for col in cols:
    print('--------------------------')
    print(f'Executing wrapper for {col}!')
    main(col, lr = lr_dict[col], epochs = epoch_dict[col], unfreeze_all = tuning[col])
submit.head()
submit.to_csv("predictions_roberta_large_v2.csv", index=False)
for col in cols:
    print(submit[col].sum()/submit.shape[0])
for col in cols:
    print(train[col].sum()/train.shape[0])
