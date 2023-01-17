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
df = pd.read_csv('../input/mushrooms.csv', ',')
print(df.columns)
print(df['class'].unique())
feats = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

feat_cardinality = dict((feat, len(df[feat].unique())) for feat in feats)
feat_cardinality
def generate_sentence(row):
    return " ".join(
        ["{feat_name}:{feat_val}".format(feat_name=feat_name, feat_val=row[feat_name]) for feat_name in feats]
    )

df["sentence_like"] = df.apply(generate_sentence, axis=1)
def generate_target(row):
    return 1 if row["class"] == 'p' else 0

df["target"] = df.apply(generate_target, axis=1)
from sklearn.model_selection import train_test_split

## split to train and val
train_df, val_df = train_test_split(df, test_size=0.3, random_state=2018)

## Get the features
train_X = train_df["sentence_like"].values
val_X = val_df["sentence_like"].values

## Get the target values
train_y = train_df['target'].values
val_y = val_df['target'].values
def get_feature_mapping(X):
    feat_to_idx = {}
    idx_to_feat = {}
    next_idx = 0
    for line in X:
        for word in line.split(' '):
            if word not in feat_to_idx:
                idx_to_feat[next_idx] = word
                feat_to_idx[word] = next_idx
                next_idx += 1
    return idx_to_feat, feat_to_idx

idx_to_feat, feat_to_idx = get_feature_mapping(train_X)

def get_feats(X, feat_to_idx):
    num_feats = max(feat_to_idx.values()) + 1
    F = np.zeros((len(X), num_feats))
    for i, line in enumerate(X):
        for word in line.split(' '):
            if word in feat_to_idx:
                F[i, feat_to_idx[word]] = 1.
    return F

train_X = get_feats(train_X, feat_to_idx)
val_X = get_feats(val_X, feat_to_idx)
    
num_feats = max(feat_to_idx.values()) + 1 # plus one because feature indexing starts from 0 
print(num_feats)
from keras.layers import Dense, Input, Dropout
from keras.models import Model


inp = Input(shape=(num_feats,))
x = Dense(256, activation="relu")(inp)
x = Dropout(0.1)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_y, batch_size=512, epochs=10, validation_data=(val_X, val_y))
from sklearn import metrics
pred_val_y = model.predict([val_X], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.5, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y>thresh).astype(int))))
from sklearn.svm import LinearSVC
model = LinearSVC(random_state=0, tol=1e-5)
model.fit(train_X, train_y)
pred_val_y_svm = model.predict(val_X)
for thresh in np.arange(0.1, 0.5, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(val_y, (pred_val_y_svm>thresh).astype(int))))
