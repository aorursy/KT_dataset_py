import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/mbti_1.csv')

df.head()
def var_row(row):

    l = []

    for i in row.split('|||'):

        l.append(len(i.split()))

    return np.var(l)



df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)

df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))

df.head()
plt.figure(figsize=(15,10))

sns.swarmplot("type", "words_per_comment", data=df)
df.groupby('type').agg({'type':'count'})

df_2 = df[~df['type'].isin(['ESFJ','ESFP','ESTJ','ESTP'])]

df_2['http_per_comment'] = df_2['posts'].apply(lambda x: x.count('http')/50)

df_2['qm_per_comment'] = df_2['posts'].apply(lambda x: x.count('?')/50)

df_2.head()
print(df_2.groupby('type').agg({'http_per_comment': 'mean'}))

print(df_2.groupby('type').agg({'qm_per_comment': 'mean'}))
plt.figure(figsize=(15,10))

sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_2, kind="hex")
def plot_jointplot(mbti_type, axs, titles):

    df_3 = df_2[df_2['type'] == mbti_type]

    sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_3, kind="hex", ax = axs, title = titles)

    

i = df_2['type'].unique()

k = 0

for m in range(0,2):

    for n in range(0,6):

        df_3 = df_2[df_2['type'] == i[k]]

        sns.jointplot("variance_of_word_counts", "words_per_comment", data=df_3, kind="hex")

        plt.title(i[k])

        k+=1

    

from scipy.misc import imread

from wordcloud import WordCloud, STOPWORDS



fig, ax = plt.subplots(len(df['type'].unique()), sharex=True, figsize=(15,10*len(df['type'].unique())))



k = 0

for i in df['type'].unique():

    df_4 = df[df['type'] == i]

    wordcloud = WordCloud().generate(df_4['posts'].to_string())

    ax[k].imshow(wordcloud)

    ax[k].set_title(i)

    ax[k].axis("off")

    k+=1
