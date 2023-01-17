import numpy as np
import pandas as pd
import os
import re

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()
f = open('../input/wikitext2/wiki.train.tokens', "rt")
text = f.readlines()
def convertToList(text):
    res = []
    for i in range(0,len(text)):

        lis = list(text[i].split(' '))
        res.append(lis)
    return res

text = convertToList(text)
text[1]
def preprocessing(text):
    
    
    processed_text = []
    
    for tokens in tqdm(text):
        
        # remove other non-alphabets symbols with space (i.e. keep only alphabets and whitespaces).
        processed = re.sub('[^a-zA-Z ]', '', tokens)
        
        words = processed.split()
        
        # keep words that have length of more than 1 (e.g. gb, bb), remove those with length 1.
        processed_text.append(' '.join([word for word in words if len(word) > 1]))
    
    return processed_text
text_string=''

for i in text:
    for j in i:
        text_string += j
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline


def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='black', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(text_string)
plot_cloud(wordcloud)

wordcloud.to_file("wordcloud.png")

w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=4-1)
w2v_model.build_vocab(text, progress_per=10000)

# Total number of vocab in our custom word embedding

len(w2v_model.wv.vocab.keys())
# Dimension of each word (we set it to 300 in the above training step)

w2v_model.wv.vector_size

w2v_model.train(text, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
# Checking out how 'team' is represented (an array of 300 numbers)

w2v_model.wv.get_vector('team')
from sklearn.manifold import TSNE
def tsne_plot(model):
    
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(w2v_model)
def getSimilarWords(word,model):
    print(model.wv.most_similar(word))

user_input = input('Enter word to find words similar to it \n')
getSimilarWords(user_input,w2v_model)
getSimilarWords('California',w2v_model)
getSimilarWords('music',w2v_model)
from gensim.models import KeyedVectors
w2v_model.wv.save_word2vec_format('custom_wikitext_embedding.txt')
loaded_embeddings = KeyedVectors.load_word2vec_format('custom_wikitext_embedding.txt', binary=True)
