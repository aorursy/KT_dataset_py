import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/mbti_1.csv')
def var_row(row):

    l = []

    for i in row.split('|||'):

        l.append(len(i.split()))

    return np.var(l)



df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)

df['variance_of_word_counts'] = df['posts'].apply(lambda x: var_row(x))

df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)

df['qm_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)

df['INTJMENTIONS_per_comment'] = df['posts'].apply(lambda x: x.count('INTJ')/50)

df['ENFPMENTIONS_per_comment'] = df['posts'].apply(lambda x: x.count('ENFP')/50)

df.head()
a4_dims = (9, 6)

fig, ax = plt.subplots(figsize=a4_dims)

sns.countplot(ax=ax,x=df['type'])

plt.show()
df.hist(column='words_per_comment',by = "type", grid=False, bins=20

        ,xlabelsize=8, ylabelsize=8,figsize = (15,15)

       ) #

plt.show()
#df_2 = df[~df['type'].isin(['ESFJ','ESFP','ESTJ','ESTP'])]

df.hist(column='qm_per_comment',by = "type", grid=False, bins=20

        ,xlabelsize=10, ylabelsize=8,figsize = (15,15), sharex=True

       ) #

plt.show()
Means = df.groupby(['type'], as_index=False).mean()

Means = pd.DataFrame(Means)

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(ax=ax, x=df['type'], y=df['qm_per_comment'])

plt.show()
df.hist(column='http_per_comment',by = "type", grid=False, bins=20

        ,xlabelsize=10, ylabelsize=8,figsize = (15,15), sharex=True

       ) #

plt.show()
Means = df.groupby(['type'], as_index=False).mean()

Means = pd.DataFrame(Means)

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(ax=ax, x=df['type'], y=df['http_per_comment'])

plt.show()
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(ax=ax, x=df['type'], y=df['INTJMENTIONS_per_comment'])

plt.show()
a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)

sns.barplot(ax=ax, x=df['type'], y=df['ENFPMENTIONS_per_comment'])

plt.show()
#plt.figure(figsize=(15,10))

#sns.jointplot("INTJMENTIONS_per_comment", "ENFPMENTIONS_per_comment", data=df, kind="hex")

#df1 = df.ix[:,["INTJMENTIONS_per_comment", "ENFPMENTIONS_per_comment","qm_per_comment","http_per_comment"]]

#sns.heatmap(df1)



#print("? per coment (link)")

#sns.jointplot("INTJMENTIONS_per_comment", "qm_per_comment", data=df, kind="hex")

#print("HTTP per coment (link)")

#sns.jointplot("INTJMENTIONS_per_comment", "http_per_comment", data=df, kind="hex")




