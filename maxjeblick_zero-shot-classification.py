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
!pip install git+https://github.com/huggingface/transformers
from transformers import pipeline

classifier = pipeline("zero-shot-classification", device=0)
df_test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')

df_test['comment_text'] = df_test['comment_text'].apply(lambda x: ' '.join(x.split(' ')[:64]))



df_test.head()
df_sample_submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

df_sample_submission.head()
from tqdm import tqdm



def batch(iterable, batch_size=8):

    l = len(iterable)

    for ndx in tqdm(range(0, l, batch_size)):

        yield iterable[ndx:min(ndx + batch_size, l)]
comment_texts = df_test['comment_text'].values



candidate_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

hypothesis_template = "The sentiment is {}."





preds = np.array([]).reshape((-1, len(candidate_labels)))

for comment_texts_batch in batch(comment_texts):

    preds_dict = classifier(comment_texts_batch,

                            candidate_labels,

                            hypothesis_template=hypothesis_template,

                            multi_class=True

                           )

    preds_batch = np.array([pred_dict['scores'] for pred_dict in preds_dict])

    preds = np.concatenate([preds, preds_batch], axis=0)
for i, col in enumerate(candidate_labels):

    df_sample_submission[col] = preds[:, i]
df_sample_submission.to_csv('submission.csv', index=False)
df_sample_submission.head()