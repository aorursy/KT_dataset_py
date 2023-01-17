#load libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import re



from nltk.sentiment.vader import SentimentIntensityAnalyzer
#load datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.columns
bad_tags = train.iloc[:, 2:].sum()



rowsums=train.iloc[:,2:].sum(axis=1)



train['clean']= (rowsums == 0)

binary = {True : 1, False : 0} #0 - bad comments, 1 - clean comments

train["clean"] = train["clean"].map(binary)
x=train.iloc[:,2:].sum()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)
train.sample(10)
train["comment_length"] = train.comment_text.str.len()



g = sns.FacetGrid(train, hue ="clean")

g.map(sns.distplot, "comment_length")

g.fig.set_size_inches(12, 6)





plt.legend()
#Clean Comment Sample



train[train["clean"] == 1].sample(10)
#Toxic Comment Sample



train[train["toxic"] == 1].sample(10)
#Sample severe_toxic comments



train[train["severe_toxic"] == 1].sample(10)
#Sample obscene comments



train[train["obscene"] == 1].sample(10)
#Sample threat comments



train[train["threat"] == 1].sample(10)
#Sample insult comments



train[train["insult"] == 1].sample(10)
#Sample identity_hate comments



train[train["identity_hate"] == 1].sample(10)
plt.figure(figsize = (12, 8))

sns.heatmap(train.iloc[:, 2:-1].corr(), annot = True)
sentiment = SentimentIntensityAnalyzer()
toxic_vs_clean = []



for index in train.index:

    toxic_vs_clean.append(sentiment.polarity_scores(train.iloc[index, 1]))
data = pd.concat([pd.DataFrame(toxic_vs_clean), train["clean"]], axis =1)
g = sns.FacetGrid(data, hue = "clean")

g.map(sns.distplot, "compound")

g.fig.set_size_inches(12, 6)



plt.legend()
analyze = pd.DataFrame(toxic_vs_clean)



analyze["neu_neg"] = analyze["neu"]/(analyze["neg"] + 0.0001)

analyze["neu_pos"] = analyze["neu"]/(analyze["pos"] + 0.0001)



eda = pd.concat([analyze, train["clean"]], axis =1)
fig, [ax1, ax2] = plt.subplots(ncols = 2, nrows = 1, figsize = (12, 6))



sns.regplot("neu_neg", "compound", data = eda, ax = ax1)

sns.regplot("neu_pos", "compound", data = eda, ax = ax2)