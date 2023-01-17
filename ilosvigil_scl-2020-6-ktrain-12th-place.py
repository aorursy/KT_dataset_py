!pip install ktrain
import os

import random

import gc



import numpy as np

import pandas as pd

import ktrain
!pip freeze > requirements.txt
print('Numpy version:', np.__version__)

print('Pandas version:', pd.__version__)

print('ktrain version:', ktrain.__version__)
SEED = 42



os.environ['PYTHONHASHSEED']=str(SEED)

random.seed(SEED)

np.random.seed(SEED)
!lscpu
!free -m
!nvidia-smi
df_train = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/train.csv')

df_train
df_train2 = pd.read_csv('/kaggle/input/shopee-reviews/shopee_reviews.csv')

df_train2 = df_train2[df_train2['label'] != 'label']

df_train2
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

df_test
X_train = pd.concat([df_train['review'], df_train2['text']], axis=0).reset_index(drop=True)

X_test = df_test['review']

y_train = pd.concat([df_train['rating'], df_train2['label']], axis=0).reset_index(drop=True)
t = ktrain.text.Transformer('distilroberta-base', maxlen=65, classes=[str(r) for r in range(1, 6)])
y_train = y_train.apply(lambda r: str(r))



# to fix this issue https://github.com/huggingface/transformers/issues/3809

X_train = X_train.replace({'': '.'})

X_test = X_test.replace({'': '.'})
train = t.preprocess_train(X_train.to_list(), y_train.to_list())
gc.collect()
model = t.get_classifier()
model.summary()
learner = ktrain.get_learner(model, train_data=train, batch_size=320)
# Google recommender LR : 2e-5 to 5e-5

learner.fit_onecycle(3e-4, 5)
gc.collect()
predictor = ktrain.get_predictor(learner.model, preproc=t)
y_test_pred = predictor.predict(X_test.to_list())

y_test_pred = [np.int32(y) for y in y_test_pred]
df_submission = pd.concat([pd.Series(list(range(1,60428)), name='review_id', dtype=np.int32), pd.Series(y_test_pred, name='rating')], axis=1)

df_submission.to_csv('submission_preprocess_text.csv', index=False)



df_submission
df_test = pd.read_csv('/kaggle/input/student-shopee-code-league-sentiment-analysis/test.csv')

y_test_pred2 = predictor.predict(df_test['review'].to_list())
df_submission2 = pd.concat([pd.Series(list(range(1,60428)), name='review_id', dtype=np.int32), pd.Series(y_test_pred2, name='rating')], axis=1)

df_submission2.to_csv('submission_raw_text.csv', index=False)



df_submission2
y_test_pred3 = predictor.predict(X_test.to_list(), return_proba=True)

# for i in range(len(y_test_pred3)):

#     y_test_pred3[i, 0] = y_test_pred3[i, 0] * 0.11388

#     y_test_pred3[i, 1] = y_test_pred3[i, 1] * 0.02350

#     y_test_pred3[i, 2] = y_test_pred3[i, 2] * 0.06051

#     y_test_pred3[i, 3] = y_test_pred3[i, 4] * 0.39692

#     y_test_pred3[i, 4] = y_test_pred3[i, 3] * 0.40519

y_test_pred3 = np.argmax(y_test_pred3, axis=1)

for i in range(len(y_test_pred3)):

    y_test_pred3[i] = y_test_pred3[i] + 1

y_test_pred3 = [np.int32(y) for y in y_test_pred]
df_submission = pd.concat([pd.Series(list(range(1,60428)), name='review_id', dtype=np.int32), pd.Series(y_test_pred3, name='rating')], axis=1)

df_submission.to_csv('submission_preprocess_text_mod_proba.csv', index=False)



df_submission