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
!wget https://github.com/facebookresearch/fastText/archive/v0.9.2.zip
!unzip v0.9.2.zip
!cd fastText-0.9.2 && make && pip install .
train = pd.read_csv('/kaggle/input/ml-guild-classification-task/train.csv')

train_text = pd.read_csv('/kaggle/input/ml-guild-classification-task/train.csv')['review']

test_text = pd.read_csv('/kaggle/input/ml-guild-classification-task/test.csv')['review']







def process_target(row):

    pos, neg, neut = '__label__positive', '__label__negative', '__label__neutral'

    return row['negative'] * neg + row['neutral'] * neut + row['positive'] * pos





target = train.apply(lambda row: process_target(row), axis=1)

import re



def delete_garbage_chars(string):

    regex = re.compile('[^a-zA-Z0-9а-яА-Я\ ]')

    return regex.sub('', string)



train_text = train_text.apply(lambda row: ' '.join(row.lower().strip().split()))

train_text = train_text.apply(delete_garbage_chars)

test_text = test_text.apply(lambda row: ' '.join(row.lower().strip().split()))

test_text = test_text.apply(delete_garbage_chars)
# split data for training and validating



from sklearn.model_selection import train_test_split





X_train, X_val, y_train, y_val = train_test_split(train_text, target, test_size=0.3, stratify=target)
fastText_train = y_train + ' ' + X_train

fastText_train.to_csv('fastText_train.csv', index=False, header=None)

fastText_val = y_val + ' ' + X_val

fastText_val.to_csv('fastText_val.csv', index=False, header=None)
fastText_all_train = target + ' ' + train_text

fastText_all_train.to_csv('fastText_all_train.csv', index=False, header=None)
import fasttext



model = fasttext.train_supervised(input="fastText_train.csv", epoch=70, dim=128, ws=10, minCount=5, wordNgrams=3)

model.test("fastText_val.csv")



model = fasttext.train_supervised(input="fastText_all_train.csv", epoch=70, dim=128, ws=10, minCount=5, wordNgrams=3)
sample = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')

sample['positive'] = 0





preds = []



for text in test_text:

    preds.append(model.predict(text, k=3))



for enum, pred in enumerate(preds):

    switch = {'__label__positive': 'positive', '__label__negative': 'negative', '__label__neutral': 'neutral'}

    for label, proba in zip(pred[0], pred[1]):

        sample.loc[enum, switch[label]] = proba

    

    

sample.to_csv('without_pretrain_submission.csv', index=False)
model = fasttext.train_supervised(input="fastText_train.csv", epoch=70, dim=300, ws=10, minCount=5, wordNgrams=3, pretrainedVectors='/kaggle/input/fasttest-common-crawl-russian/cc.ru.300.vec')

model.test("fastText_val.csv")
model = fasttext.train_supervised(input="fastText_all_train.csv", epoch=70, dim=300, ws=10, minCount=5, wordNgrams=3, pretrainedVectors='/kaggle/input/fasttest-common-crawl-russian/cc.ru.300.vec')

sample = pd.read_csv('/kaggle/input/ml-guild-classification-task/sample_submission.csv')

sample['positive'] = 0





preds = []



for text in test_text:

    preds.append(model.predict(text, k=3))



for enum, pred in enumerate(preds):

    switch = {'__label__positive': 'positive', '__label__negative': 'negative', '__label__neutral': 'neutral'}

    for label, proba in zip(pred[0], pred[1]):

        sample.loc[enum, switch[label]] = proba

    

    

sample.to_csv('with_pretrain_submission.csv', index=False)
!rm -rf fastText-0.9.2/