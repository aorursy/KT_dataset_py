# Installing RAPIDS

import sys

!cp ../input/rapids/rapids.0.15.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.7/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.7"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from cuml.manifold import TSNE

import cupy, cudf

import os

import matplotlib.pyplot as plt

import gc
# As cudf is faster than pandas, I'm going to use that.

df = cudf.read_csv('../input/simple-eda-and-baseline-data-generation/train_preprocessed.csv')
df['prior_question_had_explanation'] = df['prior_question_had_explanation'].astype(int)

df = df.fillna(-9999)

df['prior_question_elapsed_time'] = df['prior_question_elapsed_time'].replace(['inf'], -9999)

df['prior_question_elapsed_time'] = df['prior_question_elapsed_time'].astype(float)
sampled_df = df.sample(10000)

del df
target = sampled_df['answered_correctly']

del sampled_df['answered_correctly']
gc.collect()
target = target.values

sampled_df = sampled_df.values
# I'm converting the data to numpy, as then it's easier to plot/save it etc.

target = cupy.asnumpy(target)

sampled_df = cupy.asnumpy(sampled_df)
%%time

tsne = TSNE(n_components=2)

tsne_data = tsne.fit_transform(sampled_df)
plt.scatter(tsne_data[:,0], tsne_data[:,1], c = target, s = 0.6)