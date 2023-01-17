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
test_df = pd.read_csv("../input/test.csv")

test_df.mean(axis=0).quantile(np.linspace(0, 1, 11))
test_df.std(axis=0).quantile(np.linspace(0, 1, 11))
num_cols = test_df.drop(["id"], axis=1).values.shape[1]

num_cols
np.random.seed(180)

coefficients = np.random.normal(size=num_cols)
def create_row():

    return np.random.normal(size=(num_cols))
syn_df = np.stack([create_row() for i in range(20_000)])
mask = np.random.binomial(1, 0.333, size=num_cols)

result = (((mask * coefficients) * syn_df).sum(axis=1))

np.min(result), np.median(result), np.max(result)
noise = np.random.normal(size=result.shape)

categories = ((result + noise) > -4)*1

categories[:10], (categories == 1).sum()/len(categories), (categories == 0).sum()/len(categories)
!mkdir -p "synthetic/"

train = syn_df[:250, :]

test = syn_df[250:, :]



train_df = pd.DataFrame(train)

train_df['target'] = categories[:250]

train_df.to_csv("train.csv", index_label="id")



test_df = pd.DataFrame(test)

test_df['target'] = categories[250:]

test_df.to_csv("test.csv", index_label="id")