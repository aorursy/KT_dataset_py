!pip install pytorch-pretrained-bert
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from fastai.text import * 

from fastai.callbacks import *

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

from pytorch_pretrained_bert import BertTokenizer

from shutil import copyfile

print(os.listdir("../input"))
#reading into pandas and renaming columns for easier api access

filepath = Path('../input/') #twitter-airline-sentiment

df = pd.read_csv(filepath/'Tweets.csv')

df.rename(columns={'airline_sentiment':'label'},inplace=True)



df = df[['label','text']]

df.head(2)
#Setting path for learner

path = Path(os.path.abspath(os.curdir))
df['neutral'] = df['label'].apply(lambda x: 1. if x=='neutral' else 0.)

df['positive'] = df['label'].apply(lambda x: 1. if x=='positive' else 0.)

df['negative'] = df['label'].apply(lambda x: 1. if x=='negative' else 0.)
#80-20 split into train/validation set.

train = df[:int(len(df)*.80)]

valid = df[int(len(df)*.80):]



train.to_csv('train.csv',index_label=False )

valid.to_csv('valid.csv',index_label=False )
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

    bert_model_name="bert-base-uncased",

    max_lr=3e-5,

    epochs=4,

    use_fp16=True,

    bs=32,

    discriminative=False,

    max_seq_len=256,

)
bert_tok = BertTokenizer.from_pretrained(config.bert_model_name)
class FastAiBertTokenizer(BaseTokenizer): 

    """Wrapper around BertTokenizer to be compatible with fast.ai"""

    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs): 

         self._pretrained_tokenizer = tokenizer 

         self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

         return self 

    def tokenizer(self, t:str) -> List[str]: #Limits the maximum sequence length

        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]        
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])
def _join_texts(texts:Collection[str], mark_fields:bool=False, sos_token:Optional[str]=BOS):

    """Borrowed from fast.ai source"""

    if not isinstance(texts, np.ndarray): texts = np.array(texts)

    if is1d(texts): texts = texts[:,None]

    df = pd.DataFrame({i:texts[:,i] for i in range(texts.shape[1])})

    text_col = f'{FLD} {1} ' + df[0].astype(str) if mark_fields else df[0].astype(str)

    if sos_token is not None: text_col = f"{sos_token} " + text_col

    for i in range(1,len(df.columns)):

        #text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i]

        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df[i].astype(str)

    return text_col.values
from sklearn.model_selection import train_test_split



train, test = [pd.read_csv(path / fname) for fname in ["train.csv", "valid.csv"]]

val = test # we won't be using a validation set but you can easily create one using train_test_split
if config.testing:

    train = train.head(1024)

    val = val.head(1024)

    test = test.head(1024)
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])
label_cols = ["negative", "neutral", "positive"]
databunch = TextDataBunch.from_df(".", train, val, test,

                   tokenizer=fastai_tokenizer,

                   vocab=fastai_bert_vocab,

                   include_bos=False,

                   include_eos=False,

                   text_cols="text",

                   label_cols=label_cols,

                   bs=config.bs,

                   collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

              )
class BertTokenizeProcessor(TokenizeProcessor):

    def __init__(self, tokenizer):

        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)



class BertNumericalizeProcessor(NumericalizeProcessor):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)



def get_bert_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):

    """

    Constructing preprocessors for BERT

    We remove sos/eos tokens since we add that ourselves in the tokenizer.

    We also use a custom vocabulary to match the numericalization with the original BERT model.

    """

    return [BertTokenizeProcessor(tokenizer=tokenizer),

            NumericalizeProcessor(vocab=vocab)]



class BertDataBunch(TextDataBunch):

    @classmethod

    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,

                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,

                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:

        "Create a `TextDataBunch` from DataFrames."

        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_bert_processor)

        # use our custom processors while taking tokenizer and vocab as kwargs

        processor = get_bert_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)

        if classes is None and is_listy(label_cols) and len(label_cols) > 1: classes = label_cols

        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor),

                        TextList.from_df(valid_df, path, cols=text_cols, processor=processor))

        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)

        if test_df is not None: src.add_test(TextList.from_df(test_df, path, cols=text_cols))

        return src.databunch(**kwargs)
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=3)
loss_func = nn.BCEWithLogitsLoss()
from fastai.callbacks import CSVLogger
learner = Learner(databunch, bert_model,loss_func=loss_func, callback_fns=[partial(CSVLogger, append=True)])

if config.use_fp16: learner = learner.to_fp16()
#learner.lr_find(); learner.recorder.plot()
#learner.fit_one_cycle(10, 4e-4)

learner.fit_one_cycle(4, 3e-5)
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    y = learner.get_preds(ds_type)[1].detach().cpu().numpy()

    

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    

    return preds[reverse_sampler, :], y[reverse_sampler, :] 
preds, y = get_preds_as_nparray(DatasetType.Valid)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score
#convert predicted prob to predicted labels

idx = np.argmax(preds, axis=-1)

y_preds = np.zeros(preds.shape)

y_preds[np.arange(preds.shape[0]), idx] = 1
accuracy_score(y_preds, y)
f1_score(y, y_preds, average='macro') 
f1_score(y, y_preds, average='micro') 
f1_score(y, y_preds, average='weighted')  
f1_score(y, y_preds, average=None)
learner.save('bert')