# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Читаем файлы
train = pd.read_csv('/kaggle/input/sf-dst-recommendation-challenge/train.csv')
test = pd.read_csv('/kaggle/input/sf-dst-recommendation-challenge/test.csv')
submission = pd.read_csv('/kaggle/input/sf-dst-recommendation-challenge/sample_submission.csv')
import scipy.sparse as sparse

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import sklearn
from sklearn.model_selection import train_test_split

import scipy.sparse as sparse

train_data, test_data = train_test_split(train,random_state=32, shuffle=True)
ratings_coo = sparse.coo_matrix((train_data['rating'].astype(int),
                                 (train_data['userid'],
                                  train_data['itemid'])))
NUM_THREADS = 4 #число потоков
NUM_COMPONENTS = 30 #число параметров вектора 
NUM_EPOCHS = 20 #число эпох обучения

model = LightFM(learning_rate=0.1, loss='logistic',
                no_components=NUM_COMPONENTS)
model = model.fit(ratings_coo, epochs=NUM_EPOCHS, 
                  num_threads=NUM_THREADS)
preds = model.predict(test_data.userid.values,
                      test_data.itemid.values)
sklearn.metrics.roc_auc_score(test_data.rating,preds)
preds = model.predict(test.userid.values,
                      test.itemid.values)
preds.min(), preds.max()
normalized_preds = (preds - preds.min())/(preds - preds.min()).max()
normalized_preds.min(), normalized_preds.max()
submission['rating']= normalized_preds