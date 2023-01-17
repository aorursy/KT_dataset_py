# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!cp '/kaggle/input/nlp-getting-started/test.csv' '/kaggle/working/test.csv'

!cp '/kaggle/input/nlp-getting-started/train.csv' '/kaggle/working/train.csv'
!pip install fastai --upgrade
%reload_ext autoreload

%autoreload 2



from fastai.text.all import *

path = Path('/kaggle/working/')

path.ls()
train_df = pd.read_csv(path/'train.csv')

test_df = pd.read_csv(path/'test.csv')

print(train_df.head())

#simple_train_df = train_df[['id', 'text', 'target']]

#simple_test_df = train_df[['id', 'text']]
dls = TextDataLoaders.from_csv(path=path, 

                               csv_fname='train.csv', 

                               text_col='text', 

                               label_col='target', 

                               valid_pct=0.2)

first(dls[0])
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=F1Score())

learn.model_dir = '/kaggle/working/'

learn.fine_tune(2, 1e-2)
learn.save('simple_NLP')
learn.load('simple_NLP')
print(learn.predict('It was crazy'))

print(learn.predict('It was a disaster'))
print('Total number of needed predictions:',len(test_df))

predictions = []

ids = []



for i in range(len(test_df)):

    print(i)

    pred, _, _ = learn.predict(test_df.loc[i, 'text'])

    ids.append(test_df.loc[i, 'id'])

    predictions.append(pred)



#print(predictions, ids)
submission = pd.DataFrame({'id': ids, 'target':predictions})

print(submission)

submission.to_csv("simple_submission.csv", index=False)
df = pd.concat([train_df, test_df])
df.head()
dls_lm = TextDataLoaders.from_df(df, text_col='text', is_lm=True, valid_pct=0.0)

dls_lm.show_batch(max_n=3)
lm_learn = language_model_learner(dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()], path=path, wd=0.1)

lm_learn.model_dir = '/kaggle/working/'
#lm_learn.lr_find()
lm_learn.fit_one_cycle(1, 1e-2)

lm_learn.save('1epoch')
lm_learn = lm_learn.load('1epoch')

lm_learn.unfreeze()

lm_learn.fit_one_cycle(10, 1e-3)

lm_learn.save_encoder('finetuned')
dls_clas = TextDataLoaders.from_csv(path=path, 

                               csv_fname='train.csv', 

                               text_col='text', 

                               label_col='target', 

                               valid_pct=0.2,

                               text_vocab=dls_lm.vocab)
adv_learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=F1Score())

adv_learn.model_dir = '/kaggle/working/'

adv_learn = adv_learn.load_encoder('finetuned')

adv_learn.fit_one_cycle(1, 2e-2)
adv_learn.freeze_to(-2)

adv_learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
adv_learn.freeze_to(-3)

adv_learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
adv_learn.unfreeze()

adv_learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
test_df.loc[0, 'text']
adv_learn.predict(test_df.loc[0, 'text'])
predictions = []

ids = []



for i in range(len(test_df)):

    print(i)

    pred, _, _ = adv_learn.predict(test_df.loc[i, 'text'])

    ids.append(test_df.loc[i, 'id'])

    predictions.append(pred)
submission = pd.DataFrame({'id': ids, 'target':predictions})

print(submission)

submission.to_csv("complex_submission.csv", index=False)
dls_lm_back = TextDataLoaders.from_df(df, text_col='text', is_lm=True, valid_pct=0, backward = True)

dls_lm_back.show_batch(max_n=3)
lm_learn_back = language_model_learner(dls_lm_back, AWD_LSTM, metrics=[accuracy, Perplexity()], path=path, wd=0.1)

lm_learn_back.model_dir = '/kaggle/working/'
lm_learn_back.fit_one_cycle(1, 1e-2)

lm_learn_back.save('1epoch_back')
lm_learn_back = lm_learn_back.load('1epoch_back')

lm_learn_back.unfreeze()

lm_learn_back.fit_one_cycle(10, 1e-3)

lm_learn_back.save_encoder('finetuned_back')
dls_clas_back = TextDataLoaders.from_csv(path=path, 

                               csv_fname='train.csv', 

                               text_col='text', 

                               label_col='target', 

                               valid_pct=0.2,

                               text_vocab=dls_lm_back.vocab,

                               backwards = True)
back_learn = text_classifier_learner(dls_clas_back, AWD_LSTM, drop_mult=0.5, metrics=F1Score())

back_learn.model_dir = '/kaggle/working/'

back_learn = back_learn.load_encoder('finetuned_back')

back_learn.fit_one_cycle(1, 2e-2)
back_learn.freeze_to(-2)

back_learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
back_learn.freeze_to(-3)

back_learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
back_learn.unfreeze()

back_learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
back_learn.predict(test_df.loc[0, 'text'])
predictions = []

ids = []



for i in range(len(test_df)):

    print(i)

    _, _, forward_predict = adv_learn.predict(test_df.loc[i, 'text'])

    _, _, backwards_predict = back_learn.predict(test_df.loc[i, 'text'])

    pred = bool(forward_predict[1]+backwards_predict[1] > 1)

    ids.append(test_df.loc[i, 'id'])

    predictions.append(int(pred))
submission = pd.DataFrame({'id': ids, 'target':predictions})

print(submission)

submission.to_csv("ensemble_submission.csv", index=False)