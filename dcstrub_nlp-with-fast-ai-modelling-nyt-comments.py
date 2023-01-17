# %load_ext autotime ## missing and cannot install anything in GPU instance
import os

import numpy as np

import pandas as pd

import re

import torch

import fastai.text as ait

import altair as alt
# alt.renderers.enable('notebook')

alt.data_transformers.enable('json')
bs = 48
dir_nyt = '/kaggle/input/nyt-comments'
for dirname, _, filenames in os.walk(dir_nyt):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_data_sample = pd.read_csv(os.path.join(dir_nyt, 'CommentsApril2018.csv'))
df_data_sample.columns
df_data_sample.sample(2)
columns = ["commentID", "commentBody", "editorsSelection", "recommendations"]
df_data = pd.DataFrame()



for dirname, _, filenames in os.walk(dir_nyt):

    for filename in filenames:

        if re.search("Comment", filename):

            df_month = pd.read_csv(os.path.join(dir_nyt, filename))

            df_month = df_month[columns]

            df_data = pd.concat([df_data, df_month])
df_data.info()
df_data_sample = df_data_sample[columns]
from bs4 import BeautifulSoup
html = df_data_sample.commentText[13]

print(html)

txt = BeautifulSoup(html, "lxml").text

print(txt)
# df_data_sample.commentText = (df_data_sample.commentBody.

#                               apply(lambda html: BeautifulSoup(html, "lxml").text)

#                              )
df_data_sample.head()
p_unsupervised = 0.7

p_train = 0.67

p_test = 0.33
msk = np.random.rand(len(df_data)) < p_unsupervised

df_data_unsup = df_data[msk]

df_data_other = df_data[~msk]

msk = np.random.rand(len(df_data_other)) < p_train

df_data_train = df_data_other[msk]

df_data_test = df_data_other[~msk]
for df in [df_data_unsup, df_data_train, df_data_test]:

    print(len(df) / len(df_data))
msk = np.random.rand(len(df_data_sample)) < p_unsupervised

df_data_sample_unsup = df_data_sample[msk]

df_data_sample_other = df_data_sample[~msk]

msk = np.random.rand(len(df_data_sample_other)) < p_train

df_data_sample_train = df_data_sample_other[msk]

df_data_sample_test = df_data_sample_other[~msk]
df_plot = df_data_train.copy()

df_plot["text_length"] = df_plot["commentBody"].apply(lambda x: len(x))
(alt.Chart(df_plot)

    .mark_bar()

    .encode(

        x=alt.X('text_length', bin=True),

        y='count()'

    )

)
data = df_plot["text_length"].values

np.append(data.min(), np.percentile(data, [25, 50, 75, 100]))
df_plot[["commentID", "editorsSelection"]].groupby("editorsSelection").count()
data = df_plot["recommendations"].values

np.append(data.min(), np.percentile(data, [25, 50, 75, 100]))
data = ait.data.TextDataBunch.from_df(

    path="/kaggle/working", 

    train_df=df_data_sample_unsup,

    valid_df=df_data_sample_train,

    text_cols="commentBody",

    label_cols="editorsSelection", 

    bs=bs

)
data.vocab.itos[:25]
data.train_ds[0][0]
data.train_ds[0][0].data[:10]
data.show_batch()