import numpy as np
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import seaborn as sns

from nltk.util import ngrams
from nltk.corpus import stopwords 
from nltk import FreqDist
from collections import Counter
from wordcloud import WordCloud

plt.rcParams.update({'font.size': 22})
stop_words = set(stopwords.words('english'))
print(os.getcwd())
for path, dirs, files in os.walk("../"):
  print (path)
  for f in files:
    print ("\t{}".format(f))
df = pd.read_csv("../input/train.csv")
df.info()
print(df[(df["target"] == 0)].iloc[:5,1].values)
print("\n")
print(df[(df["target"] == 1)].iloc[:5,1].values)
strings = df['question_text'].values
words = []
allCharsCount = 0
removedCount = 0
removedInfo = []
for s in strings:
    s = s.lower()
    
    allCharsCount += len(s)
    removed = re.sub(r'(?![a-zA-Z0-9\s]).', "", s)
    dif = len(s) - len(removed)
    removedCount += dif
    removedInfo.append([dif,dif/len(s)])
    
    tokenized = re.sub(r'(?![a-zA-Z0-9\s]).', " ", s).split(" ")
    words += [word for word in tokenized if len(word) != 0]
print("{} chars removed ({:.3}%)".format(removedCount, removedCount*100/allCharsCount))
removedInfo = pd.DataFrame(removedInfo, columns=["sum", "percent"])
sortedRemovedInfo = removedInfo.sort_values(by="percent")
print("\tMost affected sentences:\n")
for x in np.arange(-1,-11,-1):
    ind = sortedRemovedInfo.index[x]
    print("with {:.2f}% of deletions:  {}\n".format(
        sortedRemovedInfo.iloc[x,1]*100, strings[ind]))
ind1Pct = int(sortedRemovedInfo.shape[0]*0.999)
ind2Pct = int(sortedRemovedInfo.shape[0]*0.99)

fig, ax = plt.subplots(3,1,figsize=(15,15))
fig.suptitle('Distribution for: Percent of characters removed in a sentence')
ax[0].set_title("Full distribution")
ax[1].set_title("Lowest 99.9% distribution")
ax[2].set_title("Lowest 99.0% distribution")
sns.distplot(sortedRemovedInfo.iloc[:,1], ax=ax[0])
sns.distplot(sortedRemovedInfo.iloc[:ind1Pct,1], ax=ax[1])
sns.distplot(sortedRemovedInfo.iloc[:ind2Pct,1], ax=ax[2])
plt.show()
df["question_len"] = df["question_text"].apply(lambda x: len(x))
lenSorted = df.sort_values(by="question_len")
print("Shortest questions:")
lenSorted[["question_text", "target"]].iloc[:20]
vals = lenSorted.iloc[:-10]
insincere = vals[(vals["target"] == 1)]["question_len"]
sincere = vals[(vals["target"] == 0)]["question_len"]
plt.figure(figsize=(15,5))
plt.hist([insincere, sincere], stacked=True, bins = 50,
         label=["insincere", "sincere"])
plt.legend()
plt.title("Distribution of length of questions")
plt.show()
lowVals = lenSorted.iloc[:100]
insincere = lowVals[(lowVals["target"] == 1)]["question_len"]
sincere = lowVals[(lowVals["target"] == 0)]["question_len"]
plt.figure(figsize=(15,5))
plt.hist([insincere, sincere], stacked=True, bins = 12,
         label=["insincere", "sincere"])
plt.legend()
plt.title("Distribution of the shortest 100 questions")
plt.show()
lenSorted[["question_text", "target"]].iloc[-20:]
highVals = lenSorted.iloc[-100:-10]
insincere = highVals[(highVals["target"] == 1)]["question_len"]
sincere = highVals[(highVals["target"] == 0)]["question_len"]
plt.figure(figsize=(15,5))
plt.hist([insincere, sincere], stacked=True, bins = 12,
         label=["insincere", "sincere"])
plt.legend()
plt.show()
def word_analysis(tokens):
    frequency_dict = Counter(tokens)
    most_common = frequency_dict.most_common(200)
    most_common = [entry for entry in most_common if (entry[0] not in stop_words)]
    wc = WordCloud(background_color='white', width=2500, height=500)
    wc.generate_from_frequencies(dict(most_common))
    
    most_common = dict(list(most_common[:20]))
    
    fig, axes = plt.subplots(2,1, figsize=(25,10))
    axes[0].imshow(wc)
    axes[0].axis('off')
    axes[1].bar(most_common.keys(), most_common.values())
    plt.xticks(rotation=45)
plt.rcParams.update({'font.size': 22})
word_analysis(words)
bigrams = ngrams(words,2)
bigram_dist = FreqDist()
bigram_dist.update(bigrams)

plt.figure(figsize=(20,8))
bigram_dist.plot(25)
plt.show()
trigrams = ngrams(words,3)
trigram_dist = FreqDist()
trigram_dist.update(trigrams)

plt.figure(figsize=(20,8))
trigram_dist.plot(25)
plt.show()






















