!pip install fastai2
from fastai2.text.all import *
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')

# print the size of the training and test data
print(train.shape, test.shape)
train.head(2)
test.head(2)
total = pd.concat((train.drop('target', axis=1),test), axis=0)
total.shape
total.head()
dls_lm = TextDataLoaders.from_df(total, path='.', valid_pct=0.1, is_lm=True, text_col ='text')
dls_lm.show_batch()
learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()])
learn.lr_find()
learn.fit_one_cycle(1, 1e-1)
learn.save('1epoch')
learn = learn.load('1epoch')
learn.unfreeze()
learn.fit_one_cycle(6, 2e-3)
learn.save_encoder('finetuned')
dls_clas = TextDataLoaders.from_df(train, path='.', text_col='text', label_col='target', valid_pct=0.1, text_vocab=dls_lm.train.vocab)
dls_clas.show_batch(max_n=4)
len(dls_clas.train.vocab[0]), len(dls_lm.train.vocab)
learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn = learn.load_encoder('finetuned')
learn.fit_one_cycle(2, 2e-2)
learn.freeze_to(-2)
learn.fit_one_cycle(2, slice(1e-2/(2.6**4),1e-2))
learn.freeze_to(-3)
learn.fit_one_cycle(3, slice(5e-3/(2.6**4),5e-3))
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-3))
test_dl = dls_clas.test_dl(test)
preds, _, classes = learn.get_preds(dl=test_dl, with_decoded=True)
df = pd.DataFrame({
    'id': test_dl.get_idxs(),
    'target': classes
})
df.head()
df = df.sort_values(by='id')
df = df.reset_index(drop=True)
df.id = test.id.values
df.head()
df.to_csv('submission.csv', index=False)
