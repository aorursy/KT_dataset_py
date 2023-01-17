import numpy as np

import pandas as pd

from fastai.text import *
bs = 24

path = Path('../input/student-shopee-code-league-sentiment-analysis')
train1 = pd.read_csv('../input/shopee-reviews/shopee_reviews.csv', low_memory=False)[['text','label']]

train1.rename(columns={'text':'review', 'label':'rating'},inplace=True)

train1 = train1[train1['rating']!='label']

train1
train2 = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/train.csv').drop('review_id', axis=1)

train2
np.random.seed(42)

train3 =  train1.sample(frac=1).groupby('rating').apply(lambda x: x[:40000]).reset_index(drop=True)

train3['rating'] = train3['rating'].astype(int)

train3
train = pd.concat([train2,train3], ignore_index=True)

train
test = pd.read_csv('../input/student-shopee-code-league-sentiment-analysis/test.csv')

test
df = pd.concat([train.drop(['rating'],1),test.drop('review_id',1)], ignore_index=True)

df
np.random.seed(42)

data_lm = (TextList.from_df(df)

            .split_by_rand_pct(0.1)

            .label_for_lm()

            .databunch(bs=bs))
np.random.seed(42)

data_lm.show_batch()
np.random.seed(42)

data_clas = (TextList.from_df(train, path, vocab=data_lm.vocab)

                .split_by_rand_pct(0.5)

                .label_from_df(cols=1)

                .add_test(test['review'])

                .databunch(bs=bs))
np.random.seed(42)

data_clas.show_batch()
learn = language_model_learner(data_lm, AWD_LSTM, model_dir="/tmp/models")
learn.fit_one_cycle(1, 3e-2, moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
learn.save_encoder('fine_tuned_enc')
learn = text_classifier_learner(data_clas, AWD_LSTM, model_dir="/tmp/models")

learn.load_encoder('fine_tuned_enc')
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
learn.freeze_to(-2)

learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))
learn.freeze_to(-3)

learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
learn.unfreeze()

learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
pred_list = [int(np.argmax(i)+1) for i in learn.get_preds(ds_type=DatasetType.Test)[0]]
submit = pd.DataFrame({'review_id':test['review_id'],'rating':pred_list})

submit.to_csv("submit.csv", index=False)