INPUT_DIR = '../input/'
OUTPUT_DIR = './'
!ls -lah {INPUT_DIR}
%load_ext autoreload
%autoreload 2
%matplotlib inline

import re

import pprint
pp = pprint.PrettyPrinter(indent=2).pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.image as mpimg

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    cross_val_score, ShuffleSplit, train_test_split, GridSearchCV)
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

sns.set()
mpl.rcParams['figure.facecolor'] = 'white'
## read data
df = pd.read_csv(f'{INPUT_DIR}train.csv')
df.head()
def show_digit(df, i, preds=None, y_val=None):
    if df.shape[1] == 785:
        arr = df.iloc[i, 1:]
    else:
        arr = df.iloc[i]
    arr = arr.values.reshape((28, 28))
    fig, ax = plt.subplots(figsize=(2,2)) 
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(arr, cmap='gray')
    if df.shape[1] == 785:
        plt.title(df.iloc[i, 0])
    else:
        plt.title(
            f"{y_val.iloc[i]} (predicted {preds[i]})"
            if y_val is not None else
            f"predicted {preds[i]}"
        )
for i in range(5):
    show_digit(df, i)
x = df.iloc[:, 1:]
y = df.iloc[:, 0]
x_trn, x_val, y_trn, y_val = train_test_split(x, y, test_size=0.2)
# keep only 5k to both speed things up and prevent overfitting
# (the latter works because train_test_split does random subsampling)
x = x[:5000]
y = y[:5000]
df.iloc[3, 1:].hist()  # check out the colors distribution
# worth fixing n_estimators=10 - this is the default in current sklearn
# version, but will change to 100 and maybe that could overfit
m0 = RandomForestClassifier(n_estimators=10, oob_score=True)
m0
m0.fit(x_trn, y_trn)
print("OOB score:", m0.oob_score_)
print("test score:", m0.score(x_trn, y_trn))
# NOTE: ignore warnings, they're from OOB not being always computable
print("validation score:", m0.score(x_val, y_val))
preds = m0.predict(x_val)
for i in range(10):
    show_digit(x_val, i, preds, y_val)
df_test = pd.read_csv(f'{INPUT_DIR}test.csv')
df_test.head()
test_preds = m0.predict(df_test)
for i in range(10):
    show_digit(df_test, i, test_preds)
test_tesult = pd.DataFrame({
    'ImageId': df_test.index.values + 1,
    'Label': test_preds
})
display(test_tesult.head())
test_tesult.to_csv(f'{OUTPUT_DIR}results_simple_rf.csv', index=False)