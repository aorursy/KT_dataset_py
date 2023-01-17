#import libraries

import numpy as np 
import pandas as pd
from fastai.text import * 
base_path="../output"
text_columns=['text']
label_columns=['target']
bs=32
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
            .databunch(bs=bs))
data_lm.save('tmp_lm')
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(10, 1e-2)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)

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
learn.fit_one_cycle(4, slice(5e-3/2., 5e-3))
learn.save('stage2')
learn.export()
learn.save('final')
from fastai.vision import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(6,6), dpi=60)
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
