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



from catboost import CatBoostClassifier, Pool
df = pd.read_csv('../input/cat-in-the-dat/train.csv')

df = df.drop(columns=['id'])



target = df['target'].values.tolist()



df = df.drop(columns=['target'])

dataset = df.values.tolist()



train_size = int(len(dataset))

train_x = dataset[:train_size]

train_y = target[:train_size]

test_x = dataset[train_size:]

test_y = target[train_size:]



cat_cols = []



for i in range(len(train_x[0])):

    if type(train_x[0][i]).__name__=='str':

        cat_cols.append(i)

model = CatBoostClassifier(iterations=2000, 

                           task_type="GPU",

                           loss_function = "Logloss",

                           devices='0:1')

model.fit(train_x,

          train_y,

          verbose=False, cat_features=cat_cols,plot=True)
test_df = pd.read_csv('../input/cat-in-the-dat/test.csv')

test_index = test_df.id

test_df = test_df.drop(columns=['id'])

test_dataset = test_df.values.tolist()
test_proba = model.predict_proba(test_dataset)[:,1]
submission = pd.DataFrame({'id': test_index, 'target': test_proba})

submission.to_csv('submission.csv', index=False)