# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn import preprocessing

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import tensorflow as tf

from tensorflow.keras import layers, optimizers, metrics, losses, models
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

sample = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')
test.shape
test.loc[:,'target'] = -1
data = pd.concat([train, test]).reset_index(drop=True)
data.shape
features = [f for f in train.columns if f not in ['id', 'target']]
features
for feat in features:

    lbl_enc = preprocessing.LabelEncoder()

    data.loc[:, feat] = lbl_enc.fit_transform(data[feat].astype(str).fillna('-1').values)
data.head()
train = data[data.target != -1].reset_index(drop=True)

test = data[data.target == -1].reset_index(drop=True)
test.drop(['target'], axis=1, inplace=True)
def get_model(df, cat_cols):

    inputs = []

    outputs = []

    for c in cat_cols:

        num_unique = int(df[c].nunique())

        embed_dim = int(min(np.ceil(num_unique / 2), 50))

        inp = layers.Input(shape=(1,))

        out = layers.Embedding(num_unique+2000, embed_dim, name=c)(inp)

        out = layers.Reshape(target_shape=(embed_dim,))(out)

        inputs.append(inp)

        outputs.append(out)

    

    x = layers.Concatenate()(outputs)

    x = layers.Dense(300, activation='relu')(x)

    x = layers.Dropout(0.3)(x)

    y = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=y)

    return model
get_model(train, features).summary()
model = get_model(train, features)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit([train.loc[:, f].values for f in features], train.target.values, epochs=10, batch_size=64)
predictions = model.predict([test.loc[:, f].values for f in features], batch_size=64)
sample['target'] = predictions 
sample.head()
sample.to_csv('entity_embedding.csv', index=False)