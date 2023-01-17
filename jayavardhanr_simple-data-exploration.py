#simple Exploration of The Dataset
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

from wordcloud import WordCloud



import warnings

warnings.filterwarnings('ignore')
train_df=pd.read_csv('../input/train.csv')
train_df.head(20)
train_df.shape
test_df=pd.read_csv('../input/test.csv')
#if any toxic label was set to 1

train_df['toxicity_label']=(train_df==1).any(axis=1)
is_toxic = train_df['toxicity_label'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_toxic.index, is_toxic.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Toxicity label', fontsize=12)

plt.show()
colormap = plt.cm.viridis

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation plot of toxic behaviors', y=1.05, size=15)

sns.heatmap(train_df[['toxicity_label','toxic','severe_toxic','obscene','threat','insult','identity_hate']].astype(float).corr(),linewidths=0.1,

            vmax=1.0, square=True, cmap=colormap, linecolor='white',

            annot=True)
is_toxic = train_df['toxic'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_toxic.index, is_toxic.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Toxic', fontsize=12)

plt.show()
is_severe_toxic = train_df['severe_toxic'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_severe_toxic.index, is_severe_toxic.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Severely Toxic', fontsize=12)

plt.show()
is_obscene = train_df['obscene'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_obscene.index, is_obscene.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Obscene', fontsize=12)

plt.show()
is_threat = train_df['threat'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_threat.index, is_threat.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Threat', fontsize=12)

plt.show()
is_insult = train_df['insult'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_insult.index, is_insult.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Insult', fontsize=12)

plt.show()
is_identity_hate = train_df['identity_hate'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(is_identity_hate.index, is_identity_hate.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Identity Hate', fontsize=12)

plt.show()
train_qs = pd.Series(train_df['comment_text'].tolist()).astype(str)

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
test_qs = pd.Series(test_df['comment_text'].tolist()).astype(str)

cloud = WordCloud(width=1440, height=1080).generate(" ".join(test_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')