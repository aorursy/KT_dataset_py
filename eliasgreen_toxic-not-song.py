import numpy as np

import pandas as pd



from fastai.text import *



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_pure_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

validation_pure_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_valid_translated.csv")

test_pure_data = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv")
train_pure_data.head(2)
validation_pure_data.head(2)
test_pure_data.head(2)
test_pure_data['content'] = test_pure_data['translated']

test_pure_data.drop(['translated', 'id'], axis=1, inplace=True)
test_pure_data.head(2)
validation_pure_data['comment_text'] = validation_pure_data['translated']

validation_pure_data.drop(['translated', 'id'], axis=1, inplace=True)
validation_pure_data.head(2)
validation_pure_data.drop(['lang'], inplace=True, axis=1)

test_pure_data.drop(['lang'], inplace=True, axis=1)
validation_pure_data.head(2)
test_pure_data.head(2)
train_pure_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis = 0)
train_pure_data['toxic'] = train_pure_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum(axis = 1) > 0
train_pure_data.drop(['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'id'], inplace=True, axis=1)
train_pure_data['toxic'] = train_pure_data['toxic'].astype(int)
train_pure_data.head(2)
train_positive_samples = train_pure_data[train_pure_data['toxic'] == 1]

train_negative_samples = train_pure_data[train_pure_data['toxic'] == 0]
#final_train = pd.concat([train_positive_samples, train_negative_samples.sample(24000, random_state=3543)])

final_train = train_pure_data.sample(frac=1, random_state=3543)
final_train.head()
data_lm = (TextList.from_df(final_train)

                   .split_by_rand_pct()

                   .label_from_df(cols='toxic')

                   .databunch())



data_lm.save()
data_lm.show_batch()
learn = text_classifier_learner(data_lm, AWD_LSTM)

learn.unfreeze()
learn.fit_one_cycle(10, slice(1e-7, 1e-1))

#learn.save('mini_train_clas')
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)

interp.plot_confusion_matrix()
learn.data.add_test(test_pure_data)

preds,y = learn.get_preds(ds_type=DatasetType.Test)
submission = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv")
submission.head()
submission['toxic'] = [x[1].item() for x in preds]
submission.head()
submission.to_csv('submission.csv', index=False)