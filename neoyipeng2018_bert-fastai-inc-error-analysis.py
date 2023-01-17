!pip install pytorch-pretrained-bert
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from shutil import copyfile



from fastai.text import * 

from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification

from pytorch_pretrained_bert import BertTokenizer



from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score
#Setting path for learner

path = Path(os.path.abspath(os.curdir))
# Create directory

dirName = 'models'

 

try:

    # Create target Directory

    os.mkdir(dirName)

    print("Directory:" , dirName , "created!") 

except FileExistsError:

    print("Directory:" , dirName , "already exists!")
#copying files into working path

modelpath = Path('../input/bert-fastai-error-analysis')



copyfile(modelpath/"models/bert-1.pth", path/"models/bert-1.pth")
#reading into pandas and renaming columns for easier api access

filepath = Path('../input/quora-insincere-questions-classification')

trn = pd.read_csv(filepath/'train.csv')
trn.rename(columns={'target':'label', 'question_text':'text'},inplace=True)

df = trn[['label','text']]



df['1'] = df['label'].apply(lambda x: 1 if x==1 else 0)

df['0'] = df['label'].apply(lambda x: 1 if x==0 else 0)



valid = df[int(len(df)*.80):]
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
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])
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
fastai_bert_vocab = Vocab(list(bert_tok.vocab.keys()))
fastai_tokenizer = Tokenizer(tok_func=FastAiBertTokenizer(bert_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])
label_cols = ["1", "0"]
databunch = TextDataBunch.from_df(".", valid, valid, valid,

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
bert_model = BertForSequenceClassification.from_pretrained(config.bert_model_name, num_labels=2)
learner = Learner(databunch, bert_model, loss_func=nn.BCEWithLogitsLoss())

if config.use_fp16: learner = learner.to_fp16()
learner.load('bert-1')
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
#testing on bottom 20% of data

preds, y = get_preds_as_nparray(DatasetType.Valid)
valid.reset_index(inplace=True)

valid.rename(columns={'1':'insincere_gt', '0': 'sincere_gt'}, inplace=True); print(valid.shape); valid.head(2)
preds_df = pd.DataFrame(preds,columns=['insincere_pred', 'sincere_pred']) ; print(preds_df.shape) ; preds_df.head(2)
#convert predicted prob to predicted labels

idx = np.argmax(preds, axis=-1)

y_preds = np.zeros(preds.shape)

y_preds[np.arange(preds.shape[0]), idx] = 1



accuracy_score(y_preds, y)
y_proba = preds[:, 0]

y_finalpred = np.asarray([1 if x>0.3 else 0 for x in y_proba ])
label_df = pd.DataFrame(y_finalpred,columns=['label_pred']) ; print(label_df.shape) ; label_df.head(2)
final = valid.merge(preds_df, left_index=True, right_index=True)

final = final.merge(label_df, left_index=True, right_index=True)

final.head(2)
final.to_csv('final.csv')
#change y from multi-label to single label

idx = np.argmax(y, axis=-1)

y_new = np.zeros((y.shape[0],1))

for i in range(len(idx)):

    if idx[i]==0:y_new[i]=1
accuracy_score(y_finalpred, y_new)
f1_score(y_new, y_finalpred)