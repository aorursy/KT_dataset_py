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
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

df.head()
import seaborn as sns



sns.countplot(df.Class)
pos_features = df[df['Class'] == 1].iloc[:, :-1]

neg_features = df[df['Class'] == 0].iloc[:, :-1]



pos_labels = df[df['Class'] == 1].iloc[:, -1]

neg_labels = df[df['Class'] == 0].iloc[:, -1]
import tensorflow as tf



BUFFER_SIZE = 100000



def make_ds(features, labels):

    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()

    ds = ds.shuffle(BUFFER_SIZE).repeat()

    

    return ds



pos_ds = make_ds(pos_features, pos_labels)

neg_ds = make_ds(neg_features, neg_labels)
for features, label in pos_ds.take(1):

    print("Features:\n", features.numpy())

    print()

    print("Label: ", label.numpy())
BATCH_SIZE = 32



resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])

resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(100)
for i, (batch_data, batch_label) in enumerate(resampled_ds.take(5)):

    print(f"batch count: {str(i)}")

    print(f"batch dataset shape: {batch_data.shape}")

    print(f"pos label: {np.sum(batch_label == 1)}\nneg label: {np.sum(batch_label == 0)}")

    print("")