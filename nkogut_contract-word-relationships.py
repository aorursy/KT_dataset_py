!pip install python-docx
import numpy as np 
import pandas as pd 
import os
import docx
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
import re  
from gensim.models import word2vec 
import nltk  
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  
from nltk.stem.porter import *  
lemmatizer = WordNetLemmatizer()
%matplotlib inline
  

all_doc = []
fullText = []
fullText_str = []

def getText(filename):
    fullText = []
    doc = docx.Document(filename)
    for para in doc.paragraphs:
        para.text = para.text.replace(u'\t', u'')
        para.text = para.text.replace(u'\xa0', u'')
        fullText.append(para.text)
    add_to_alldoc(fullText)
    fullText = []

def add_to_alldoc(_fullText):
    fullText_str = []
    fullText_str = ' '.join(map(str, _fullText))
    all_doc.append(fullText_str)
    fullText_str = []
     
def review_to_words(raw_review):   # clean text 
    
    letters_only = re.sub("[^а-яА-Я]", " ", raw_review)   
    words = letters_only.lower().split()   
    stops = set(stopwords.words("russian"))  
    meaningful_words = [w for w in words if not w in stops]     #Remove stop words    
    singles = [lemmatizer.lemmatize(word) for word in meaningful_words]  
    
    return( " ".join( singles ))   
  
def build_corpus(data):  #"Creates a list of lists containing words from each sentence"  
    corpus = []  
    for sentence in data:  
        word_list = sentence.split(" ")  
        corpus.append(word_list)   

    return corpus    

def tsne_plot(model):  
    #"Creates a TSNE model and plots it"  
    labels = []  
    tokens = []  

    for word in model.wv.vocab:  
        tokens.append(model[word])  
        labels.append(word)  

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500)  
    new_values = tsne_model.fit_transform(tokens)  

    x = []  
    y = []  
    for value in new_values:  
        x.append(value[0])  
        y.append(value[1])  

    plt.figure(figsize=(8, 8))   
    
    for i in range(len(x)):  
        plt.scatter(x[i],y[i])  
        plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')  
    plt.show()  

def main():
    for r,d,f in os.walk('/kaggle/input/dogovora'):
        for file in f:
            if file.endswith(".docx"):
                getText(os.path.join(r, file))
    processed_text = [review_to_words(text) for text in all_doc] 
    corpus = build_corpus(processed_text)  
    
    model = word2vec.Word2Vec(corpus, size=100, window=5, min_count=1000, workers=4) 
#     print([x for x in model.wv.vocab][0:5])
#     print([(item[0],round(item[1],2)) for item in model.most_similar('услуг')])
    tsne_plot(model)  
    
main()
