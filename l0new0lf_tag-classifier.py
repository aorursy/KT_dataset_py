import os

import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip")

df.head()
df.info()
# get duplicates

df_dups = df[df.duplicated(['Title', 'Body', 'Tags'])]

print('Total Duplicates: ', len(df_dups))

print('ratio: ', len(df_dups)/len(df))
# remove duplicates

df = df.drop_duplicates(['Title', 'Body', 'Tags'])

print('After removing dups: ', len(df))

print('ratio: ', len(df)/6034194)
x = df["Tags"].apply(lambda x: type(x)==float)

x[x==True]
# removing `Tags` which are float instead of str

df.drop([err_idx for err_idx in x[x==True].index], inplace=True)
df["num_of_tags"] = df["Tags"].apply(lambda x: len(x.split(" ")))

df['num_of_tags'].value_counts()
plt.close()



plt.bar(

    df.num_of_tags.value_counts().index,

    df.num_of_tags.value_counts()

)



plt.xlabel('Number of tags')

plt.ylabel('Freq (x10^6)')

plt.show()
from sklearn.feature_extraction.text import CountVectorizer



# get unique tags w/ help of BoW. Tags are space separated

vectorizer = CountVectorizer(tokenizer = lambda x: x.split())



# fit_transform

# - learn the vocabulary and store in `vectorizer`

# - convert training data into feature vectors

#    - converts each input (tag) into one hot encoded based on vocab

tag_vecs = vectorizer.fit_transform(df['Tags'])
# learnt vocabulary

vocab = vectorizer.get_feature_names()

print(vocab[:5])



# total vocabulary

print('Total vocabulary: ', len(vocab))
# one hot encoded training data

print('Num of samples: ', tag_vecs.shape[0])

print('Size of one hot encoded vec (each val represents a tag): ', tag_vecs.shape[1])
# distribution of unique tags

freq_of_tags = tag_vecs.sum(axis=0).getA1() # (1, vocab_size) -> (vocab_size) i.e flatten it

tags = vocab



tag_freq = zip(tags[:5], freq_of_tags[:5])



for tag, freq in tag_freq:

    print(tag, ':', freq)
sorted_idxs = np.argsort(- freq_of_tags) # -1: descending



sorted_freqs = freq_of_tags[sorted_idxs] 

sorted_tags  = np.array(tags)[sorted_idxs]



for tag, freq in zip(sorted_tags[:5], sorted_freqs[:5]):

    print(tag, ':', freq)
# distribution of occurances

plt.close()



plt.plot(sorted_freqs)



plt.title("Distribution of number of times tag appeared questions\n")

plt.grid()

plt.xlabel("Tag idx in vocabulary")

plt.ylabel("Number of times tag appeared")

plt.show()
# zoom in first 1k

plt.close()



plt.plot(sorted_freqs[:1000])



plt.title("Distribution of number of times tag appeared questions\n")

plt.grid()

plt.xlabel("Tag idx in vocabulary")

plt.ylabel("Number of times tag appeared")

plt.show()
# zoom in first 200

plt.close()



plt.plot(sorted_freqs[:200])



plt.title("Distribution of number of times tag appeared questions\n")

plt.grid()

plt.xlabel("Tag idx in vocabulary")

plt.ylabel("Number of times tag appeared")

plt.show()
# zoom in first 100

plt.close()



# quantiles with 0.05 difference

plt.scatter(x=list(range(0,100,5)), y=sorted_freqs[0:100:5], c='orange', label="quantiles with 0.05 intervals")

# quantiles with 0.25 difference

plt.scatter(x=list(range(0,100,25)), y=sorted_freqs[0:100:25], c='m', label = "quantiles with 0.25 intervals")



for x,y in zip(list(range(0,100,25)), sorted_freqs[0:100:25]):

    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))

    

#for x,y in zip(list(range(0,100,5)), sorted_freqs[0:100:5]):

#    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))



x=100

y=sorted_freqs[100]

plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))





plt.plot(sorted_freqs[:100])



plt.legend()

plt.grid()



plt.title("Distribution of top 100 tags\n")

plt.xlabel("Tag idx in vocabulary")

plt.ylabel("Number of times tag appeared")

plt.show()





# ---------------------------------------------------------------------

# PDF AND CDF

plt.close()

plt.figure(figsize=(10,10))



plt.subplot(211)

counts, bin_edges = np.histogram(sorted_freqs, bins=100, 

                                 density = True)

pdf = counts/(sum(counts))

#print(pdf);

#print(bin_edges)

cdf = np.cumsum(pdf)



plt.title("CDF all tags\n")

plt.xlabel("Freq of tag occurances")

plt.ylabel("Percent of Tags out of all tags")

plt.grid()



plt.plot(bin_edges[1:], cdf)



# -------------

plt.subplot(212)

counts, bin_edges = np.histogram(sorted_freqs[:100], bins=100, 

                                 density = True)

pdf = counts/(sum(counts))

#print(pdf);

#print(bin_edges)

cdf = np.cumsum(pdf)



#plt.title("CDF top 100 tags\n")

plt.xlabel("Freq of to 100 tag occurances")

plt.ylabel("Percent of Tags ut of 100 tags")

plt.grid()



plt.plot(bin_edges[1:], cdf)



plt.show()
# visulaize all tags wrt their frequencies

from wordcloud import WordCloud



# input is (tag, fre) tuple

tup = dict(zip(sorted_tags, sorted_freqs))



#Initializing WordCloud using frequencies of tags.

wordcloud = WordCloud(    background_color='black',

                          width=1600,

                          height=800,

                    ).generate_from_frequencies(tup)



fig = plt.figure(figsize=(30,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad=0)

fig.savefig("tag.png")

plt.show()
plt.close()

plt.figure(figsize=(20, 5))



plt.bar(sorted_tags[:20], sorted_freqs[:20])



plt.xlabel('Top 20 Tags')

plt.ylabel('Counts')

plt.show()