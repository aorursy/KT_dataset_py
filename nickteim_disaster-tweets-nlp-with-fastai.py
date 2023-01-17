from fastai.text import *
path=Path('/kaggle/input/nlp-getting-started/')

train_df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv',index_col='id')

train_df.head(5)
cols=['text']

data_bunch = (TextList.from_df(train_df, cols=cols)

                .split_by_rand_pct(0.2)

                .label_for_lm()  

                .databunch(bs=48))

data_bunch.show_batch()
learn = language_model_learner(data_bunch,

                               AWD_LSTM,

                               pretrained_fnames=['/kaggle/input/wt103-fastai-nlp/lstm_fwd','/kaggle/input/wt103-fastai-nlp/itos_wt103'],

                               pretrained=True,

                               drop_mult=0.5)

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8, 1e-2)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, slice(1e-05, 1e-03))
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, slice(1e-06, 1e-03))
learn.freeze()
learn.fit_one_cycle(1, slice(1e-04, 1e-03))
learn.save_encoder('fine_tuned_enc')
train=train_df[:8000]

val=train_df[2000:]

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv',index_col='id')
target_cols=['target']

data_clas = TextClasDataBunch.from_df('.', train, val, test,

                  vocab=data_bunch.vocab,

                  text_cols=cols,

                  label_cols=target_cols,

                  bs=32)
learn_classifier = text_classifier_learner(data_clas, 

                                           AWD_LSTM,

                                           pretrained=False,

                                           drop_mult=0.8,

                                           metrics=[accuracy])





fnames = ['/kaggle/input/wt103-fastai-nlp/lstm_fwd.pth','/kaggle/input/wt103-fastai-nlp/itos_wt103.pkl']

learn_classifier.load_pretrained(*fnames, strict=False)







# load the trained model without target from encoder saved  

learn_classifier.load_encoder('fine_tuned_enc')
learn_classifier.lr_find()

learn_classifier.recorder.plot()
learn_classifier.fit_one_cycle(4, 1e-3)
learn_classifier.fit_one_cycle(2, 1e-05)
learn_classifier.fit_one_cycle(7, slice(1e-06, 1e-03))
preds_test, target_test = learn_classifier.get_preds(DatasetType.Test, ordered=True)

y = torch.argmax(preds_test, dim=1)

y.numpy().shape
submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

print(submission.shape)

submission['target']=y.numpy()

submission.head()
submission['target'].value_counts()
submission.to_csv('submission.csv',index=False)

print('Model ready for submission!')