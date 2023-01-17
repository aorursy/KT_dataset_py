import os
from collections import Counter
import math
import nltk # for removing stopwords from word set
from nltk.corpus import stopwords
nltk.download("stopwords") # this is a stopword dictionary 
corpus = {} # corpus dictionary
num_documents = 0
total_words = {} # dictionary for words count

dir = "../input"
os.chdir(dir)

for file in os.listdir(dir):
    num_documents += 1
    
    with open(file, "r", encoding="utf-8-sig") as input_a:
        words = []
        for line in input_a:
            line = line.lower()
            line = line.strip().split()
            words += line
            
    filtered_word = []
    num_words = 0
    
    for word in words:
        num_words += 1
        if not word in stopwords.words("english"):
            filtered_word.append(word)
               
    total_words[file] = num_words
    corpus[file] = filtered_word # corpus dictionary
total_words
def tf_counter(corpus):
    term_frequency = {}
    for text in corpus.keys():
        term_frequency[text] = Counter(corpus[text])
    return(term_frequency)
tfs = tf_counter(corpus)
tfs
def idf_calculator(term_frequency):
    
    n = len(corpus.keys()) # total number of documents in the corpus
    wordlist = []
    idfs = {}

    for text in term_frequency.keys():
        for word in term_frequency[text].keys():
            wordlist.append(word)
            
    indocu_frequency = Counter(wordlist)
    
    for word, count in indocu_frequency.items():
        idfs[word] = math.log( n / count )
        
    return idfs
idfs = idf_calculator(tfs)
idfs