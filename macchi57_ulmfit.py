%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.text import *
# bs=48

# bs=24

bs=192
torch.cuda.set_device(0)
import pandas as pd
%%capture

train = pd.read_csv('../input/fakenewsvortexbsb/train_df.csv', sep=';');
train.head()
train.shape
train.groupby('Class').count()
train = train.rename(columns={'Class': 'label', 'manchete': 'text'})
del train['index']
import os

path = os.getcwd()
%%capture

train2 = pd.read_csv('../input/news-of-the-site-folhauol/articles.csv');
train2.head(10)
train2.groupby('category').count()
train2 = train2.loc[train2['category'].isin(['poder','mundo','mercado'])].reset_index(drop=True)
train2.head(10)
train2.shape
del train2['text']

del train2['date']

del train2['subcategory']

del train2['link']

del train2['category']
train2 = train2.dropna()
from sklearn.model_selection import train_test_split

train_ml, test_ml = train_test_split( train2, test_size=0.10, random_state=42)
data_lm = TextLMDataBunch.from_df('./', train_df=train_ml, valid_df=test_ml, text_cols=0)
from sklearn.model_selection import train_test_split

train_c, test_c = train_test_split( train, test_size=0.25, random_state=42)
%%capture

valid = pd.read_csv('../input/fakenewsvortexbsb/sample_submission.csv', sep=';');
valid.head()
index = valid['index']
valid.shape
valid = valid.rename(columns={ 'Manchete': 'text'})

del valid['index']

valid= valid.dropna()
train_c.head()
test_c.head()
valid.head()
data_clas = TextDataBunch.from_df('./', train_df=train_c , valid_df=test_c , test_df=valid, vocab=data_lm.train_ds.vocab, bs=32, text_cols=0, label_cols=1)
data_lm.vocab.itos[:10]
data_lm.train_ds[0][0]
data_lm.train_ds[0][0].data[:10]
data_lm.show_batch()
data_clas.show_batch()
len(data_lm.vocab.itos),len(data_lm.train_ds)
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn_lm.lr_find()
learn_lm.recorder.plot(skip_end=15)
lr = 1e-3

lr *= bs/48
learn_lm.to_fp16();
learn_lm.fit_one_cycle(10, lr*10, moms=(0.8,0.7))
learn_lm.save('fit_1')
learn_lm.unfreeze()
learn_lm.fit_one_cycle(10, lr, moms=(0.8,0.7))
learn_lm.save('fine_tuned')
learn_lm.save_encoder('fine_tuned_enc')
enc = learn_lm.model[0].encoder
TEXT = "Bolsonaro"

N_WORDS = 10

N_SENTENCES = 2
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))
bs=48
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3).to_fp16()

learn_c.load_encoder('fine_tuned_enc')

learn_c.freeze()
learn_c.lr_find()
learn_c.recorder.plot()
learn_c.fit_one_cycle(1, 1e-3, moms=(0.8,0.7))
learn_c.save('first')
learn_c.freeze_to(-2)

learn_c.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
preds, targets = learn_c.get_preds(ds_type=DatasetType.Valid)



predictions = np.argmax(preds, axis = 1)

pd.crosstab(predictions, targets)
from sklearn.metrics import accuracy_score, classification_report

accuracy_score(predictions, targets)
print (classification_report(predictions, targets))
preds, targets = learn_c.get_preds(ds_type=DatasetType.Test, ordered=True)



predictions = np.argmax(preds, axis = 1)

pd.crosstab(predictions, targets)
preds, _ = learn_c.get_preds(ds_type=DatasetType.Test, ordered=True)

pred_prob, pred_class = preds.max(1)
pred_class
minha_sub = pd.DataFrame({'index':index,'Category':pred_class})
minha_sub
minha_sub.to_csv('submission.csv', index=False)