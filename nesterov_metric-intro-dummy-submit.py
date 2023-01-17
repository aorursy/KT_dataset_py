# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 12
import pandas as pd
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 300)
pd.options.display.float_format = '{:,.6f}'.format
import numpy as np
from tqdm import tqdm_notebook

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_target_train.csv')
df_target_train.shape
# count most popular labels
from collections import defaultdict
r = defaultdict(int)
for _, row in df_target_train.iterrows():
    for x in row['track:genres'].split(' '):
        r[int(x)] += 1
# top-5
sorted([(k, v) for (k, v) in r.items()], key=lambda t: t[-1], reverse=True)[:10]
# all labels
labels = list(sorted(r.keys()))
# convert strings to list of ints
y_true = df_target_train['track:genres'].apply(lambda s: [int(x) for x in s.split(' ')])

y_true[:10]
# convert lists of predictions to one hot encoding
y_true = y_true.apply(lambda r: [int(x in set(r)) for x in labels])

y_true[:5]
prediction_15or38 = [int(x == 15 or x == 38) for x in labels]
# mean F1 on Train

from sklearn.metrics import f1_score

np.mean([f1_score(y, prediction_15or38) for y in tqdm_notebook(y_true)])
df_target_train = pd.read_csv('/kaggle/input/mlclass-dubai-by-ods-lecture-5-hw/df_sample_submit.csv')

df_target_train.head()
df_target_train['track:genres'] = '15 38'

df_target_train.head()
df_target_train.to_csv('./submit_all_38_15.csv', index=False)



