import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
ted_df = pd.read_csv('../input/ted_main.csv')
transcript_df = pd.read_csv('../input/transcripts.csv')
ted_df.head(3)
transcript_df.head(3)
from nltk.stem import PorterStemmer #for word stemming
from nltk.stem import WordNetLemmatizer #for word lemmatizing
from nltk import pos_tag #for word lemmatizing
from nltk.corpus import wordnet, stopwords #for word subject analyzing and stopwords removal
from nltk.tokenize import sent_tokenize, word_tokenize #for tokenizing
from string import punctuation

import tqdm
#https://github.com/tqdm/tqdm #processing bar module
import gc
import re
def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def lemmatize_sentence(sentence):
    '''
    read sentences from the dataset and return lemmatized_sentences
    '''
    lemmatizer = WordNetLemmatizer()
    #There are some sentences linked together by the dot, we should separate them
    sentence = re.sub(r'[.](?=\w)', '. ', sentence)
    nltk_tagged = pos_tag(word_tokenize(sentence))
    #stop words: all punctuation and common stop words (https://gist.github.com/sebleier/554280)
    stop_words = set(stopwords.words('english') + list(punctuation))
    #update some into the stop_words set
    stop_words.update(['us','ve','nt','re','ll','wo','ca','m','s','t','``','...','-','—',' ','laughter','applause', 'ok', 'oh'])
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        #word after word_tokenize would be like: I'm => [I, 'm]
        #so it's necessary to remove "'" to make it possible to match the words with the stop_words set
        word = word.replace("'", "")
        #remove stop words
        if word.lower() in stop_words: continue
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)
for i in tqdm.tqdm(range(len(transcript_df['transcript']))):
    #check attribute first
    sentence = transcript_df.iloc[i, 0]
    #do word lemmatizing
    sentence = lemmatize_sentence(sentence)
    transcript_df.iloc[i, 0] = sentence
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(use_idf=True,
                        ngram_range=(1,1), # considering only 1-grams
                        min_df = 0.007,     # cut words present in less than 0.7% of documents
                        max_df = 0.07)      # cut words present in more than 7% of documents 
tfidf = vectorizer.fit_transform(transcript_df['transcript'])
print(tfidf.shape)
# Let's make a function to call the top ranked words in a vectorizer
def rank_words(terms, feature_matrix):
    sums = feature_matrix.sum(axis=0)
    data = []
    for col, term in enumerate(terms):
        data.append( (term, sums[0,col]) )
    ranked = pd.DataFrame(data, columns=['term','rank']).sort_values('rank', ascending=False)
    return ranked

ranked = rank_words(terms=vectorizer.get_feature_names(), feature_matrix=tfidf)
ranked.head()
# Let's visualize a word cloud with the frequencies obtained by idf transformation
dic = {ranked.loc[i,'term'].upper(): ranked.loc[i,'rank'] for i in range(0,len(ranked))}

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',
                      max_words=100,
                      colormap='Reds').generate_from_frequencies(dic)
fig = plt.figure(1,figsize=(12,10))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()
### Get Similarity Scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim_unigram=cosine_similarity(tfidf)
#create a column to record the vid number (index)
transcript_df['index'] = transcript_df.index.astype(str)
def get_similar_articles(x):
    return ",".join(transcript_df['index'].loc[x.argsort()[-5:-1]])
transcript_df['similar_articles_unigram']=[get_similar_articles(x) for x in sim_unigram]
for url in transcript_df.iloc[[0,663,730,1233,338], 1]:
    print(url)
','.join(re.findall('(?<=\'id\': )\d+',ted_df['related_talks'][0]))
for url in transcript_df.iloc[[0,865,1738,2276,892,1232],1]:
    print(url)