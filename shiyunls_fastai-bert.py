!pip install pytorch-transformers

# 注意打开GPU和网络
import pandas as pd

import numpy as np

from fastai.text import *
df = pd.read_excel('/kaggle/input/.xlsx')
df['x'] = df['x'].apply(lambda x: str(x).replace(' ', ''))
df.head()
df['y'].value_counts()     

 
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=.1, random_state=2)

train, valid = train_test_split(train, test_size=.1, random_state=2)
from pytorch_transformers import BertTokenizer, BertForSequenceClassification



bert_model = "bert-base-chinese"

max_seq_len = 256

batch_size = 32



bert_tokenizer = BertTokenizer.from_pretrained(bert_model)  # 这里注意要给kernel联网，否则会得到一个空对象
bert_vocab = Vocab(list(bert_tokenizer.vocab.keys()))
class BertFastaiTokenizer(BaseTokenizer):

    def __init__(self, tokenizer, max_seq_len=128, **kwargs):

        self.pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len



    def __call__(self, *args, **kwargs):

        return self



    def tokenizer(self, t):

        return ["[CLS]"] + self.pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
tok_func = BertFastaiTokenizer(bert_tokenizer, max_seq_len=max_seq_len)
bert_fastai_tokenizer = Tokenizer(

    tok_func=tok_func,

    pre_rules = [],

    post_rules = []

)
path = Path(".")
databunch = TextClasDataBunch.from_df(path, train, valid, test,

                  tokenizer=bert_fastai_tokenizer,

                  vocab=bert_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols="x",

                  label_cols='y',

                  bs=batch_size,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
# databunch.show_batch()
class MyNoTupleModel(BertForSequenceClassification):

    def forward(self, *args, **kwargs):

        return super().forward(*args, **kwargs)[0]
bert_pretrained_model = MyNoTupleModel.from_pretrained(bert_model, num_labels=12)
loss_func = nn.CrossEntropyLoss()
learn = Learner(databunch, 

                bert_pretrained_model,

                loss_func=loss_func,

                metrics=accuracy)
# learn.lr_find()
# learn.recorder.plot()
learn.fit_one_cycle(2, 1e-4)

learn.save('two_cycle_model')



# # 已经完成了对于模型的训练，并且已经存储好

# learn.load('/kaggle/working/models/two_cycle_model')
def dumb_series_prediction(n):

    preds = []

    for loc in range(n):

        preds.append(int(learn.predict(test.iloc[loc]['x'])[1]))

    return preds
preds = dumb_series_prediction(len(test))
preds[:10]
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(test.y, preds))
print(confusion_matrix(test.y, preds))