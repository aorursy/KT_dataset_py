# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
! unzip '/kaggle/input/medicalnotes-2019/data_split.zip'
# Any results you write to the current directory are saved as output.

TextList = [] #(filename, label, text)
path = 'dataset/train-data/'
import re
for label in os.listdir(path):
    for filename in os.listdir(path+label+'/'):
        file = path+label+'/'+filename
        try:
            with open(file, 'r') as f:
                text = f.readlines()
                parsetext = [] # list containing sentences
                for sentence in text:
                    if sentence == '\n':
                        pass
                    else:
                        for badcharacter in ['<B>\w{2}:', 'CC:', '<B>', '</B>', '\n']:
                            sentence = re.sub(badcharacter, '', sentence)
                        sentence = ''.join([s.lower() for s in sentence])
                        
                        parsetext.append(sentence)

            TextList.append((filename.strip('.txt'), label.strip(path), ''.join(parsetext)))
        except:
            pass

data = pd.DataFrame(TextList, columns = ['filename', 'label', 'text'])  
len(data)
data.iloc[120]
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
ts = tokenizer.tokenize(data.iloc[120]['text'])
### see that Bert tokenizer is doing
for t in ts:
    print(t, end = ' ')
## to explore the frequency of words, let's tokenize all sentences in each paragraph using BERT tokenizer, 
## concatnate them back to sentence, and use the CountVectorizer from sklearn
sentences = data['text'].values.tolist()
labels = data['label'].values.tolist()

LabelList = []
SentenceList = []
from sklearn.feature_extraction.text import CountVectorizer
for sentence, label in zip(sentences, labels):

    encoded_sentence= tokenizer.tokenize(sentence) # tokenize using Bert tokenizer
    LabelList.append(label)
    encoded_sentence = ' '.join(encoded_sentence) # merge them back to sentence
    SentenceList.append(encoded_sentence)
    
## Here we count the occurence of each tokenizer. 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(SentenceList).todense()
columns = vectorizer.get_feature_names()
X.shape
import pandas as pd
summarytable = pd.DataFrame(X, columns = columns)
summarytable['Label'] = LabelList
def findtopwords(df, label ='all', fromNth = 0, toNth = 1):
    if label !='all':
        assert label in df.Label.values, print('Label not found')
        df = summarytable[summarytable['Label']==label]
    return df.iloc[:, :-1].sum().reset_index(name= 'count').sort_values('count', ascending = False)[fromNth:toNth]

### Find top words 
for label in summarytable['Label'].unique():
    print(f'Common words for {label}')
    df = findtopwords(summarytable, label = label, fromNth=20, toNth=30)
    display(df)
### Load required packages
import torch
import torch.optim as optim
import random 

# fastai
from fastai import *
from fastai.text import *
from fastai.callbacks import *

# transformer
from transformers import BertModel, BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig
from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

### sklearn 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# split the data into train (that will be further splitted into train and val set), and a hold out test set
train, test = train_test_split(data, shuffle = True, stratify = data['label'], test_size = 0.25)
### choose model to work with
MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}
model_type = 'bert'
pretrained_model_name='bert-base-uncased'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
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

### customize fastai tokenizer
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)
fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])

### customize fastai processor, including tokenize_processor, numericalize_processor
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer) #get vocabulary
numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab) #numericalize
tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False) 

transformer_processor = [tokenize_processor, numericalize_processor]
pad_first = bool(model_type in ['xlnet'])
pad_idx = transformer_tokenizer.pad_token_id
databunch = (TextList.from_df(train, cols='text', processor=transformer_processor)
             .split_by_rand_pct(0.2)
             .label_from_df(cols= 'label')
             .add_test(test)
             .databunch(bs=16, pad_first=pad_first, pad_idx=pad_idx))
print('[CLS] token :', transformer_tokenizer.cls_token)
print('[SEP] token :', transformer_tokenizer.sep_token)
print('[PAD] token :', transformer_tokenizer.pad_token)
databunch.show_batch()
print('[CLS] id :', transformer_tokenizer.cls_token_id)
print('[SEP] id :', transformer_tokenizer.sep_token_id)
print('[PAD] id :', pad_idx)
test_one_batch = databunch.one_batch()[0]
print('Batch shape : ',test_one_batch.shape)
print(test_one_batch)
torch.cuda.is_available()
# defining our model architecture 
class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_model: PreTrainedModel):
        super(CustomTransformerModel,self).__init__()
        self.transformer = transformer_model
        
    def forward(self, input_ids, attention_mask=None):
        #attention_mask = (input_ids!=1).type(input_ids.type()) # Test attention_mask for RoBERTa
        logits = self.transformer(input_ids, attention_mask = attention_mask)[0]   
        return logits
use_fp16 = False
config = config_class.from_pretrained(pretrained_model_name)
config.num_labels = 5
config.use_bfloat16 = use_fp16
print(config)
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)
custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)

from fastai.callbacks import *
from transformers import AdamW
from functools import partial

CustomAdamW = partial(AdamW, correct_bias=False)

learner = Learner(databunch, 
                  custom_transformer_model, 
                  opt_func = CustomAdamW, 
                  metrics=[accuracy, error_rate])

# Show graph of learner stats and metrics after each epoch.
learner.callbacks.append(ShowGraph(learner))

# Put learn in FP16 precision mode. --> Seems to not working
if use_fp16: learner = learner.to_fp16()
learner.save('untrain')
learner.load('untrain');
learner.model
list_layers = [learner.model.transformer.bert.embeddings,
              learner.model.transformer.bert.encoder.layer[0],
              learner.model.transformer.bert.encoder.layer[1],
              learner.model.transformer.bert.encoder.layer[2],
              learner.model.transformer.bert.encoder.layer[3],
              learner.model.transformer.bert.encoder.layer[4],
              learner.model.transformer.bert.encoder.layer[5],
              learner.model.transformer.bert.encoder.layer[6],
              learner.model.transformer.bert.encoder.layer[7],
              learner.model.transformer.bert.encoder.layer[8],
              learner.model.transformer.bert.encoder.layer[9],
              learner.model.transformer.bert.encoder.layer[10],
              learner.model.transformer.bert.encoder.layer[11],
              learner.model.transformer.bert.pooler]
learner.split(list_layers)
num_groups = len(learner.layer_groups)
print('Learner split in',num_groups,'groups')
print(learner.layer_groups)
learner.freeze_to(-1)
learner.summary()

learner.lr_find()
learner.recorder.plot(skip_end=10,suggestion=True)
learner.fit_one_cycle(1,max_lr=3e-04,moms=(0.8,0.7))
learner.fit_one_cycle(1,max_lr=3e-04,moms=(0.8,0.7))
learner.save('first_cycle')
learner.load('first_cycle')
learner.freeze_to(-2)
learner.summary()
learner.lr_find()
learner.recorder.plot(skip_end=10,suggestion=True)
lr = 4e-5
num_groups = len(learner.layer_groups)
learner.fit_one_cycle(5, max_lr=slice(lr*0.95**num_groups, lr), moms=(0.8, 0.9))
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]

#test_preds = get_preds_as_nparray(DatasetType.Test)

test_preds = get_preds_as_nparray(DatasetType.Test)
classList = databunch.classes
test['predict'] = np.argmax(test_preds,axis=1)
test['target'] = test['label'].apply(lambda x: classList.index(x))
test
test[:3]
confusion_matrix(test['predict'], test['target'])
a = classification_report(test['predict'], test['target'])
print(a)


train[:3]
## Create a TextDataBunch suitable for training a language model.
## All the texts in the datasets are concatenated and the labels are ignored. Instead, the target is the next word in the sentence
train_df , valid_df = train_test_split(train, shuffle = True, stratify = train['label'], test_size = 0.10) # only use 10% here as we focus more on training
databunchLM = (TextLMDataBunch.from_df('.', train_df , valid_df, 
                                       text_cols = 'text', label_cols = 'label')) # as we need to specify train_df and valid_df, a trick is to slide data 
databunchLM.save()
databunchLM.show_batch()
len(databunchLM.vocab.itos) #all words in the data 8888
# transformer, transformerXL
learner = language_model_learner(databunchLM, AWD_LSTM, drop_mult=0.3,pretrained=True)
learner.model
learner.summary()
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(2, 3e-02,callbacks=[SaveModelCallback(learner, name="best_lm")], moms=(0.8,0.7))
learner.save('fit_head')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(3,1e-05,callbacks=[SaveModelCallback(learner, name="best_lm")], moms=(0.8,0.7))
learner.load('best_lm')
learner.save_encoder('enc')
len(databunchLM.train_ds.vocab.itos)
### set up data for classificaiton
databunchCLS  = TextClasDataBunch.from_df('.', 
                                       train_df=train_df,valid_df=valid_df, test_df = test,
                                       text_cols='text',label_cols='label',vocab=databunchLM.train_ds.vocab)

#TransferLearner = text_classifier_learner(databunchCLS, AWD_LSTM, drop_mult=0.3)
TransferLearner = text_classifier_learner(databunchCLS, AWD_LSTM, drop_mult=0.3)
TransferLearner.model
TransferLearner.load_encoder('enc')
TransferLearner.lr_find()
TransferLearner.recorder.plot(suggestion=True)
TransferLearner.fit_one_cycle(3,2e-02, moms=(0.8,0.7))
TransferLearner.freeze_to(-2)
TransferLearner.lr_find()
TransferLearner.recorder.plot(suggestion=True)
#TransferLearner.fit_one_cycle(3, best_clf_lr)
TransferLearner.fit_one_cycle(3,2e-03, moms=(0.8,0.7))
TransferLearner.unfreeze()
TransferLearner.lr_find()
TransferLearner.recorder.plot(suggestion=True)
TransferLearner.fit_one_cycle(3,1e-03, moms=(0.8,0.7))
classList = databunchCLS.classes
classList
print(len(test))
test[:3]
true, pred = [], []
for index, row in test.iterrows(): 
    tr = classList.index(row[1])
    p = TransferLearner.predict(row[2])[1].item()
    pred.append(p)
    true.append(tr)

confusion_matrix(true, pred)
a = classification_report(true, pred)
print(a)





