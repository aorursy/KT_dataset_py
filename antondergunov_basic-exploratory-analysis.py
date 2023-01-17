import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



pd.options.display.float_format = '{:,.2f}'.format
df = pd.read_csv("../input/train.csv")
df
df.describe().transpose()
corr = df.drop(["id", "publication_id", "parent_id", "article_id"], axis=1).corr()
fig, ax = plt.subplots(figsize=(10, 10))

ax.matshow(corr)

plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr.columns)), corr.columns);
fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(20, 10))

for i, metric in enumerate(['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']):

    ax[i // 3, i % 3].set_title(metric)

    ax[i // 3, i % 3].hist(df[df[metric] > 0][metric], bins = 10)
identities = ['asian', 'atheist', 'bisexual',

    'black', 'buddhist', 'christian', 'female', 'heterosexual', 'hindu',

    'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability',

    'jewish', 'latino', 'male', 'muslim', 'other_disability',

    'other_gender', 'other_race_or_ethnicity', 'other_religion',

    'other_sexual_orientation', 'physical_disability',

    'psychiatric_or_mental_illness', 'transgender', 'white']
df_without_na = df.dropna()
fig, ax = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(20, 10))

for i, identity in enumerate(identities):

    ax[i // 6, i % 6].set_title(identity)

    df_without_na.groupby(pd.cut(df_without_na[identity], np.arange(0, 1.0, 0.1))).target.mean().plot(ax = ax[i // 6, i % 6])
fig, ax = plt.subplots(4, 6, sharex='col', sharey='row', figsize=(20, 10))

for i, identity in enumerate(identities):

    ax[i // 6, i % 6].set_title(identity)

    df_temp = df_without_na.groupby(pd.cut(df_without_na[identity], np.arange(0, 1.0, 0.1))).target

    df_temp = pd.DataFrame({'mean_values': df_temp.mean(),'std_values': df_temp.std()})

    

    x = df_temp.reset_index().index

    mean = df_temp.mean_values.values

    std = df_temp.std_values.values



    ax[i // 6, i % 6].fill_between(x, mean + std, mean - std, color='blue', alpha=0.5)

    ax[i // 6, i % 6].plot(x, mean, color='black');