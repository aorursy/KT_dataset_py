!pip install fastai2 --quiet
from fastai2.text.all import *
path = Path('../input/jigsaw-toxic-comment-classification-challenge')
from zipfile import ZipFile



with ZipFile(path/'train.csv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output/kaggle/working')

    

with ZipFile(path/'test.csv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output/kaggle/working')

    

with ZipFile(path/'test_labels.csv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output/kaggle/working')

    

with ZipFile(path/'sample_submission.csv.zip', 'r') as zip_ref:

    zip_ref.extractall('../output/kaggle/working')
path_w = Path('../output/kaggle/working')
path_w.ls()
df = pd.read_csv(path_w/'train.csv')
df.head()
blocks = (TextBlock.from_df(text_cols='comment_text', is_lm=True, res_col_name='text'))
test_df = pd.read_csv(path_w/'test.csv')
test_df.head()
text_df = pd.Series.append(df['comment_text'], test_df['comment_text'])
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
ys = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult',

       'identity_hate']
blocks = (TextBlock.from_df('comment_text', seq_len=lm_dls.seq_len, vocab=lm_dls.vocab), 

          MultiCategoryBlock(encoded=True, vocab=ys))
toxic_clas = DataBlock(blocks=blocks,

                      get_x=ColReader('text'),

                      get_y=ColReader(ys),

                      splitter=RandomSplitter())
toxic_clas.summary(df.iloc[:100])
dls = toxic_clas.dataloaders(df)
loss_func = BCEWithLogitsLossFlat(thresh=0.8)

metrics = [partial(accuracy_multi, thresh=0.8)]
learn = text_classifier_learner(dls, AWD_LSTM, metrics=metrics, loss_func=loss_func)
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
learn.freeze_to(-3)

lr /=2

learn.fit_one_cycle(1, slice(lr/(2.6**4), lr), moms=moms, wd=0.1)
learn.unfreeze()

lr /= 5

learn.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7,0.8), wd=0.1)
dl = learn.dls.test_dl(test_df['comment_text'])
preds = learn.get_preds(dl=dl)
sub = pd.read_csv(path_w/'sample_submission.csv')
sub.head()
preds[0][0].cpu().numpy()
sub[ys] = preds[0]
sub.head()
sub.to_csv('submission.csv', index=False)