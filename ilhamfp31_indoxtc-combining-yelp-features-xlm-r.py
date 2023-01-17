import numpy as np

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_x = np.concatenate([

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-1/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-2/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-3/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-4/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-5/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-6/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-7/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-8/train_text.npy', allow_pickle=True)]),

    np.array([x for x in np.load('../input/indoxtc-extracting-yelp-features-xlm-r-9/train_text.npy', allow_pickle=True)]),

                  ])



print(train_x.shape)

np.save("train_text.npy", train_x)
train_y = pd.concat([

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-1/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-2/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-3/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-4/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-5/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-6/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-7/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-8/train_label.csv'),

    pd.read_csv('../input/indoxtc-extracting-yelp-features-xlm-r-9/train_label.csv'),

])



train_y['label'].to_csv('train_label.csv', index=False, header=['label'])



print(train_y.shape)

print(train_y.label.value_counts())

train_y.head()
!ls '.'