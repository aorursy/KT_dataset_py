# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.text import * 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_path="../output"

text_columns=['text']

label_columns=['target']

BATCH_SIZE=128
train= pd.read_csv('../input/nlp-getting-started/train.csv')

test= pd.read_csv('../input/nlp-getting-started/test.csv')

train.head()
tweets = pd.concat([train[text_columns], test[text_columns]])

print(tweets.shape)
data_lm = (TextList.from_df(tweets)

           #Inputs: all the text files in path

            .split_by_rand_pct(0.15)

           #We randomly split and keep 10% for validation

            .label_for_lm()           

           #We want to do a language model so we label accordingly

            .databunch(bs=BATCH_SIZE))

data_lm.save('tmp_lm')
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
print('Model Summary:')

print(learn.layer_groups)
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, 1e-2)

learn.save('lm_fit_head')
learn.load('lm_fit_head')

learn.unfreeze()

learn.fit_one_cycle(10, 1e-3)
learn.save_encoder('ft_enc')
data_clas = (TextList.from_df(train, cols=text_columns, vocab=data_lm.vocab)

             .split_by_rand_pct(0.15)

             .label_from_df('target')

             .add_test(test[text_columns])

             .databunch(bs=BATCH_SIZE))



data_clas.save('tmp_clas')
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.freeze_to(-1)

learn.summary()
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, 1e-3)
learn.save('stage1')
learn.load('stage1')

learn.freeze_to(-2)

learn.fit_one_cycle(5, slice(5e-3/2., 5e-3))

learn.save('stage2')
learn.load('stage2')

learn.unfreeze()

learn.fit_one_cycle(5, slice(2e-3/100, 2e-3))
learn.export()

learn.save('final')
from fastai.vision import ClassificationInterpretation



interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
interp = TextClassificationInterpretation.from_learner(learn)

interp.show_top_losses(10)
learn.predict(test.loc[0,'text'])
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in learn.data.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]
test_preds = get_preds_as_nparray(DatasetType.Test)
sample_submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

sample_submission['target'] = np.argmax(test_preds, axis=1)

sample_submission.to_csv("predictions.csv", index=False, header=True)
sample_submission['target'].value_counts()