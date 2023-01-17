# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sys





import pandas as pd

import numpy as np

from keras.layers import Dense, Dropout, Activation, Dropout

from keras import Model, Input

from keras.losses import mse, mae

from keras.optimizers import adam, sgd

from keras.utils import to_categorical

import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

import os
sys.stdout.write('Let me do this real quick! \n')

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

target = train_df['target']
folds = StratifiedKFold(n_splits=30, shuffle=True, random_state=44000)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))
def model(inp = Input((200,))):

    x = Dense(200, activation='relu')(inp)

    x = Dense(30, activation='relu')(x)

    for _ in range(30):

        x = Dense(10, activation='relu')(x)

    x = Dense(1, activation='relu')(x)

    return Model(inp, x)

# from keras.callbacks import TensorBoard



model = model()

model.compile(adam(), loss='mse', metrics=['acc'])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

    print("Fold :{}".format(fold_ + 1))

    trn_data = (train_df.iloc[trn_idx][features], target.iloc[trn_idx])

    val_data = (train_df.iloc[val_idx][features], target.iloc[val_idx])

    print(trn_data[0].shape, trn_data[1].shape)

    print(val_data[0].shape, trn_data[1].shape)

    # model.summary()

    model.fit(x=trn_data[0], y=trn_data[1], batch_size=600, epochs=3,

             validation_data = [val_data[0], val_data[1]])

predictions = model.predict(test_df[features])



predictions
predictions[predictions>0.7] = 1

predictions[predictions<=0.7] = 0
sub = pd.DataFrame({"ID_code": test_df.ID_code.values})

sub["target"] = predictions

sub.to_csv('submission.csv', index=False)