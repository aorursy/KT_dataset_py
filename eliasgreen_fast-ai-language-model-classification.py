import numpy as np

import pandas as pd



from tqdm import tqdm



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



for dirname, _, filenames in os.walk('/kaggle/working'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample_submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
df_train.head(5)
df_test.head(5)
sample_submission.head(1)
print('Count of missing values in TRAIN data is', sum(df_train.isnull().sum(axis=1)))
df_train.dropna(inplace=True)
df_train.to_csv('train.csv', index=False)
print('Count of missing values in TEST data is', sum(df_test.isnull().sum(axis=1)))
df_test.to_csv('test.csv', index=False)
from fastai.text import *
data_lm = (TextList.from_csv(path='/kaggle/working', csv_name='test.csv', cols='text')

                   .split_by_rand_pct()

                   .label_for_lm()

                   .databunch())
data_lm.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM)

learn.unfreeze()



learn.fit_one_cycle(10, 1e-3)
learn.save('mini_train_lm')

learn.save_encoder('mini_train_encoder')
learn.show_results()
data_clas = (TextList.from_csv(path='/kaggle/working', csv_name='test.csv', cols='text', vocab=data_lm.vocab)

                   .split_by_rand_pct()

                   .label_from_df(cols='sentiment')

                   .databunch(bs=100))
data_clas.show_batch()
learn = text_classifier_learner(data_clas, AWD_LSTM)

learn.load_encoder('mini_train_encoder')

learn.unfreeze()



learn.fit_one_cycle(10, slice(1e-3,1e-1))
learn.save('mini_train_clas')
learn.show_results()
txt_ci = TextClassificationInterpretation.from_learner(learn)
selected_texts = []



for text in tqdm(df_test['text'], position=0):

    mask = txt_ci.intrinsic_attention(text)[1] > 0.6

    text = text.split()

    selected_text = ' '.join([x for x, y in zip(text, mask) if y == True])

    selected_texts.append(selected_text)
sample_submission['selected_text'] = selected_texts
sample_submission.head(5)
sample_submission.to_csv('submission.csv', index=False)