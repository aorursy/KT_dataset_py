from tqdm import tqdm_notebook

import torch

import fastai

from fastai.text import *

fastai.__version__
train = pd.read_csv('../input/train.csv').fillna(' ')

valid = pd.read_csv('../input/valid.csv').fillna(' ')

test = pd.read_csv('../input/test.csv').fillna(' ')
pd.concat([train['text'], valid['text'], test['text']]).to_csv(

    'unlabeled_news.csv', index=None, header=True)
pd.concat([train[['text', 'label']],valid[['text', 'label']]]).to_csv(

    'train_28k.csv', index=None, header=True)

test[['text']].to_csv('test_5k.csv', index=None, header=True)
folder = '.'

unlabeled_file = 'unlabeled_news.csv'
%%time

data_lm = TextLMDataBunch.from_csv(folder, unlabeled_file, text_cols='text')
%%time

learn = language_model_learner(data_lm, drop_mult=0.3, arch=AWD_LSTM)
%%time

learn.lr_find(start_lr = slice(10e-7, 10e-5), end_lr=slice(0.1, 10))
learn.recorder.plot(skip_end=10, suggestion=True)
best_lm_lr = learn.recorder.min_grad_lr

best_lm_lr
%%time

learn.fit_one_cycle(1, best_lm_lr)
learn.unfreeze()
%%time

learn.fit_one_cycle(1, best_lm_lr)
learn.predict('An italian man was found dead in his yard due to', n_words=200)
learn.save_encoder('clickbait_news_enc')
train_file, test_file = 'train_28k.csv', 'test_5k.csv'
data_clas = TextClasDataBunch.from_csv(path=folder, 

                                        csv_name=train_file,

                                        test=test_file,

                                        vocab=data_lm.train_ds.vocab, 

                                        bs=64,

                                        text_cols='text', 

                                        label_cols='label')
data_clas.save('ulmfit_data_clas_clickbait_news')
learn_clas = text_classifier_learner(data_clas, drop_mult=0.3, arch=AWD_LSTM)

learn_clas.load_encoder('clickbait_news_enc')
learn_clas.lr_find(start_lr=slice(10e-7, 10e-5), end_lr=slice(0.1, 10))
learn_clas.recorder.plot(skip_end=10, suggestion=True)
best_clf_lr = learn_clas.recorder.min_grad_lr

best_clf_lr
learn_clas.fit_one_cycle(1, best_clf_lr)
learn_clas.freeze_to(-2)
learn_clas.fit_one_cycle(1, best_clf_lr)
learn_clas.unfreeze()
learn_clas.fit_one_cycle(1, best_clf_lr)
learn_clas.show_results()
data_clas.add_test(test["text"])
test_preds, _ = learn_clas.get_preds(DatasetType.Test, 

                                     ordered=True)
test_pred_df = pd.DataFrame(test_preds.data.cpu().numpy(),

                            columns=['clickbait', 'news', 'other'])

ulmfit_preds = pd.Series(np.argmax(test_pred_df.values, axis=1),

                        name='label').map({0: 'clickbait', 1: 'news', 2: 'other'})

ulmfit_preds.head()
ulmfit_preds.to_csv('ulmfit_predictions.csv', index_label='id', header=True)