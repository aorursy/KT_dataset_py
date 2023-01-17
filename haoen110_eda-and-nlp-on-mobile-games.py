import os

import re # For Regular Expression 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import nltk # Natural Language Tool Kits

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk.corpus import stopwords

from string import punctuation

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud



%matplotlib inline 

%config InlineBackend.figure_format = 'retina'
df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

df = df.drop_duplicates().reset_index()

display(df.head(5))

print(df.info())
import re # For Regular Expression 

import nltk # Natural Language Tool Kits

from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

from nltk.corpus import stopwords

from string import punctuation

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk import bigrams

from nltk import trigrams
print(df['Description'][9])
# Remove useless strings and lower the words.

def rm_lower(description):

    description = re.sub(r'\\n', ' ', description)

    description = re.sub(r'\\u....', ' ', description)

    description = re.sub(r'\\x..', ' ', description)

    description = re.sub(r'http.*html', ' ', description)

    description = re.sub(r'http.*com', ' ', description)

    description = re.sub(r'www.*com', ' ', description)

    description = re.sub(r'\W+', ' ', description)

    description = re.sub(r'\d', ' ', description)

    description = ' '.join(description.split()).lower()

    return description
print(rm_lower(df['Description'][9]))
def tk(description):

    description = rm_lower(description)

    token = word_tokenize(description)

#     token = [word.lower() for word in token]

#     token = [word for word in token if word]

    stopwords_en = set(stopwords.words('english')) # set of stopwords

#     stopwords_en_withpunct = stopwords_en.union(set(punctuation))

    token = [word for word in token if word not in stopwords_en]

    token = [WordNetLemmatizer().lemmatize(word) for word in token]

#     fdist1 = fdist.most_common(5)

#     FreqDist(token)

    return token
print(tk(df['Description'][9]))
token_dict = {}

for i in range(len(df)):

    token_dict[df['Name'][i]] = tk(df['Description'][i])
for i, (k, v) in enumerate(token_dict.items()):

    if i in range(0, 1):

        print(k, v)
entire_description = []

for i in token_dict.values():

    entire_description += i



bio_tokens = bigrams(entire_description)

temp = []

for i in bio_tokens:

    temp.append(i)

bio_tokens = temp

freq_bio = FreqDist(bio_tokens)



tri_tokens = trigrams(entire_description)

temp = []

for i in tri_tokens:

    temp.append(i)

tri_tokens = temp

freq_tri = FreqDist(tri_tokens)
freq_sig = FreqDist(entire_description)

freq_sig = pd.DataFrame({'Word': list(dict(freq_sig).keys()),

                       'Count': list(dict(freq_sig).values())})



freq_bio = FreqDist(bio_tokens)

freq_bio = pd.DataFrame({'Word': list(dict(freq_bio).keys()),

                       'Count': list(dict(freq_bio).values())})



freq_tri = FreqDist(tri_tokens)

freq_tri = pd.DataFrame({'Word': list(dict(freq_tri).keys()),

                       'Count': list(dict(freq_tri).values())})
wordcloud = WordCloud(background_color=None, width=800, height=1200, mode='RGBA').generate(' '.join(entire_description))
# Draw the wordcloud graph

plt.figure(dpi=300)

plt.imshow(wordcloud)

plt.axis('off')

plt.title('Wordcloud of Description')

plt.show()

# plt.savefig('wordcloud.pgn')
plt.figure(dpi=300)

plt.grid(ls='--')

sns.barplot(x='Count', y='Word', data=freq_sig.sort_values('Count', ascending=False)[0:20], 

            palette="Set3")

plt.title('The Most Common Words in Single Grams')

plt.show()

# plt.savefig('Singlegrams.png')
plt.figure(dpi=300)

plt.grid(ls='--')

sns.barplot(x='Count', y='Word', data=freq_bio.sort_values('Count', ascending=False)[0:20], palette="Set3")

plt.title('The Most Common Words in Birams')

plt.show()

# plt.savefig('Bigrams.png')
import networkx as nx
num = 100

nw = []

for i in range(num):

    nw.append(freq_bio.sort_values('Count', ascending=False)['Word'][i])
G = nx.from_edgelist(nw)
colo = np.array(list(dict(G.degree()).values())) 

fig = plt.figure(dpi=300)

nx.draw_circular(G, with_labels=True, node_color=colo, cmap=plt.cm.twilight, node_size=80, line_color='grey',

        linewidths=0.8, width=0.8, font_size=6, alpha=0.8, font_family='fantasy')

plt.title("Top 100 Most Common Words's Network" )

plt.show()

# plt.savefig('100network.png')
plt.figure(dpi=300)

plt.grid(ls='--')

sns.barplot(x='Count', y='Word', data=freq_tri.sort_values('Count', ascending=False)[0:20], palette="Set1")

plt.title('The Most Common Words in Trigrams')

plt.show()

# plt.savefig('Trigrams.png')
effective_words = []

for i, (k, v) in enumerate(token_dict.items()):

    if i in range(len(df)):  

        effective_words.append(len(v))

df['Effective Words'] = effective_words

display(df[['Name', 'Description', 'Effective Words']].head())
plt.figure(dpi=300)

sns.barplot(x='Average User Rating', y='Effective Words', data=df, palette="Set1")

plt.title('Average User Rating Vs. Effective Words')

plt.show()
def count_tags(tags):

    noun = 0

    adj = 0

    verb = 0

    adverb = 0

    for i in tags:

        if i[1] == 'JJ':

            adj += 1

        if i[1] == 'NN':

            noun += 1

        if i[1] == 'VB':

            verb += 1

        if i[1] == 'RB':

            adverb += 1

    return [noun, adj, verb, adverb]
num_noun = []

num_adj = []

num_verb = []

num_adverb = []

for i in df['Name']:   

    tags = count_tags(nltk.pos_tag(token_dict[i]))

    num_noun.append(tags[0])

    num_adj.append(tags[1])

    num_verb.append(tags[2])

    num_adverb.append(tags[3])
df['The Proportion of N.'] = num_noun / df['Effective Words']

df['The Proportion of Adj.'] = num_adj / df['Effective Words']

df['The Proportion of V.'] = num_verb / df['Effective Words']

df['The Proportion of Adv.'] = num_adverb / df['Effective Words']
display(df[['Name', 'Effective Words', 'The Proportion of N.', 'The Proportion of Adj.', 'The Proportion of V.', 'The Proportion of Adv.']].head())
plt.figure(dpi=300)

sns.barplot(x='Average User Rating', y='The Proportion of Adj.', data=df, palette="Set1")

plt.title('Average User Rating Vs. The Number of Adj.')

plt.show()
title_token_dict = {}

for i in range(len(df)):

    title_token_dict[df['Name'][i]] = tk(df['Name'][i])
title_entire_description = []

for i in title_token_dict.values():

    title_entire_description += i



title_bio_tokens = bigrams(title_entire_description)

temp = []

for i in title_bio_tokens:

    temp.append(i)

title_bio_tokens = temp

title_freq_bio = FreqDist(title_bio_tokens)
title_freq_sig = FreqDist(title_entire_description)

title_freq_sig = pd.DataFrame({'Word': list(dict(title_freq_sig).keys()),

                       'Count': list(dict(title_freq_sig).values())})



title_freq_bio = FreqDist(title_bio_tokens)

title_freq_bio = pd.DataFrame({'Word': list(dict(title_freq_bio).keys()),

                       'Count': list(dict(title_freq_bio).values())})
plt.figure(dpi=300)

plt.grid(ls='--')

sns.barplot(x='Count', y='Word', data=title_freq_sig.sort_values('Count', ascending=False)[1:21], palette="Set3")

plt.title('The Most Common Words in Single Grams of Title')

plt.show()
plt.figure(dpi=300)

plt.grid(ls='--')

sns.barplot(x='Count', y='Word', data=title_freq_bio.sort_values('Count', ascending=False)[1:21], palette="Set3")

plt.title('The Most Common Words in Birams of Title')

plt.show()