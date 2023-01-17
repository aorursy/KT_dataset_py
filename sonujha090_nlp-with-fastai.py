!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -q
!pip install --upgrade kornia -q
!pip install allennlp==1.1.0.rc4 -q
!pip install --upgrade fastai -q
import torch
print(torch.__version__)
print(torch.cuda.is_available())

import fastai
print(fastai.__version__)

from fastai.text.all import *
path = Path('../input/nlp-getting-started')
Path.BASE_PATH = path
path.ls()
submisson = pd.read_csv(path/'sample_submission.csv')
submisson.head()
train_df = pd.read_csv(path/'train.csv')
train_df.head()
test_df = pd.read_csv(path/'test.csv')
test_df.head()
blocks = (TextBlock.from_df(text_cols='text', is_lm=True))
text_df = pd.Series.append(train_df['text'], test_df['text'])
text_df = pd.DataFrame(text_df)
text_df.head()
get_x = ColReader('text')
splitter = RandomSplitter(0.1, seed=42)
lm_dblock = DataBlock(blocks=blocks,
                     get_x=get_x,
                     splitter=splitter)
lm_dls = lm_dblock.dataloaders(text_df, bs=64)
lm_learn = language_model_learner(lm_dls, AWD_LSTM, pretrained=True, metrics=[accuracy, Perplexity()])
lm_learn.to_fp16()
lm_learn.fine_tune(10, 4e-3)
lm_learn.save_encoder('fine_tuned')
train_df.head()
blocks = (TextBlock.from_df('text', seq_len=lm_dls.seq_len, vocab=lm_dls.vocab), CategoryBlock(vocab={0,1}))
toxic_clas = DataBlock(blocks=blocks,
                      get_x=ColReader('text'),
                      get_y=ColReader('target'),
                      splitter=RandomSplitter())
dls = toxic_clas.summary(train_df[:10])
ds = toxic_clas.datasets(train_df)
ds[0]
dls = toxic_clas.dataloaders(train_df)
dls.show_batch()
xb, yb = dls.one_batch()
xb.shape , yb.shape
learn = text_classifier_learner(dls, AWD_LSTM, metrics=accuracy)
learn.lr_find()
learn.load_encoder('fine_tuned');
learn.to_fp16()

lr = 1e-2
moms = (0.8,0.7, 0.8)
lr *= learn.dls.bs/128
learn.fit_one_cycle(1, lr, moms=moms, wd=0.1)
learn.freeze_to(-2)
lr/=2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=0.1)

dl = learn.dls.test_dl(test_df['text'])
preds = learn.get_preds(dl=dl)
labels = np.argmax(preds[0], axis=1)
labels
submisson.head()
submisson['target'] = labels
submisson.head()