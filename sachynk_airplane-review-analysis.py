import pandas as pd
from nltk.corpus import wordnet
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

from nltk import ngrams, FreqDist
import math
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from IPython.display import display
from tqdm import tqdm
from collections import Counter
import ast

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sb

from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import scipy.stats as stats

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE

from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()

%matplotlib inline
df = pd.read_excel("../input/word-freq/word_freq.xlsx")
from itertools import islice

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
d={}
for a,b in df.values:
    d[a] = b
take(20, d.items())
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wordcloud = WordCloud(height= 350, width = 550)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize= (15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)
pos_reviews_df = pd.read_excel('../input/positive-reviews/positive reviews_f.xlsx')
pos_reviews_df["review_clean"] = pos_reviews_df["positive_review"].apply(lambda x: clean_text(x))
pos_reviews_df["review_clean"][1]
pos_reviews_df["positive_review"][1]
def get_top_n_words(n_top_words, count_vectorizer, text_data):
    '''
    returns a tuple of the top n words in a sample and their 
    accompanying counts, given a CountVectorizer object and text sample
    '''
    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)
    vectorized_total = np.sum(vectorized_headlines, axis=0)
    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)
    word_values = np.flip(np.sort(vectorized_total)[0,:],1)
    
    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))
    for i in range(n_top_words):
        word_vectors[i,word_indices[0,i]] = 1

    words = [word[0].encode('ascii').decode('utf-8') for 
             word in count_vectorizer.inverse_transform(word_vectors)]

    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')
words, word_values = get_top_n_words(n_top_words=50,
                                     count_vectorizer=count_vectorizer, 
                                     text_data=pos_reviews_df["review_clean"])

fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), word_values);
ax.set_xticks(range(len(words)));
ax.set_xticklabels(words, rotation='vertical');
ax.set_title('Top words in headlines dataset (excluding stop words)');
ax.set_xlabel('Word');
ax.set_ylabel('Number of occurences');
plt.show()
words, word_values = get_top_n_words(n_top_words=100,
                                     count_vectorizer=count_vectorizer, 
                                     text_data=pos_reviews_df["review_clean"])
d={}
for i in range(0,len(words)):
    d[words[i]] = word_values[i]
d
import matplotlib.pyplot as plt
from wordcloud import WordCloud


wordcloud = WordCloud(height=350, width=550)
wordcloud.generate_from_frequencies(frequencies=d)
plt.figure(figsize= (13,13))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")
plt.show()
tagged_headlines = [TextBlob(pos_reviews_df["review_clean"][i]).pos_tags for i in range(pos_reviews_df.shape[0])]
tagged_headlines_df = pd.DataFrame({'tags':tagged_headlines})
word_counts = [] 
pos_counts = {}

for headline in tagged_headlines_df[u'tags']:
    word_counts.append(len(headline))
    for tag in headline:
        if tag[1] in pos_counts:
            pos_counts[tag[1]] += 1
        else:
            pos_counts[tag[1]] = 1
            
print('Total number of words: ', np.sum(word_counts))
print('Mean number of words per headline: ', np.mean(word_counts))
pos_sorted_types = sorted(pos_counts, key=pos_counts.__getitem__, reverse=True)
pos_sorted_counts = sorted(pos_counts.values(), reverse=True)

fig, ax = plt.subplots(figsize=(18,8))
ax.bar(range(len(pos_counts)), pos_sorted_counts);
ax.set_xticks(range(len(pos_counts)));
ax.set_xticklabels(pos_sorted_types);
ax.set_title('Part-of-Speech Tagging for Headlines Corpus');
ax.set_xlabel('Type of Word');
words =pos_reviews_df["positive_review"].astype(str).str.cat(sep = " ")
words = words.split(" ")
words = [word for word in words if len(word)>1]
words = [word for word in words if not word.isnumeric()]
words = [word for word in words if word!= 'nbsp' ]

words = [word.lower() for word in words]
fDist = FreqDist(words)
for word, f in fDist.most_common(50):
    print(u"{},{},{},{} ".format(word,f, math.log(f),math.log(f)/math.log(fDist.most_common(1)[0][1])))
          
with open("word_saliency.csv", "w") as fp:
    
    fp.writelines('word, raw_freq, log_freq, saliency')
    fp.write('\n')
    for word, f in fDist.most_common():
        fp.write(u"{},{},{},{} ".format(word,f, math.log(f),math.log(f)/math.log(fDist.most_common(1)[0][1])))
        fp.write('\n')
    
