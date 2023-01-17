from IPython.display import Image
Image("../input/graphs/brands.png")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import re
import itertools
from __future__ import print_function
import pandas as pd
import datetime as dt
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import RegexpTokenizer
import string
import scipy.sparse as sparse
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


import spacy
import en_core_web_sm
from spacy import displacy
nlp = spacy.load('en')
#nlp = en_core_web_sm.load()
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
# to provide serial of numbers
def gen(n):
    while True:
        yield n
        n += 1
df = pd.read_csv("../input/panasonic/Panasonic_headphones.csv", encoding='utf-8')
df.info()
def cleaned_reviews(x):
    return(''.join(re.sub('[^a-zA-Z_.]', ' ', x)))

def get_bigram_likelihood(statements, freq_filter=3, nbest=200):
    """
    Returns n (likelihood ratio) bi-grams from a group of documents
    :param        statements: list of strings
    :param output_file: output path for saved file
    :param freq_filter: filter for # of appearances in bi-gram
    :param       nbest: likelihood ratio for bi-grams
    """

    #words = list()
    #tokenize sentence into words
    #for statement in statements:
        # remove non-words
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(statements)

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_finder = BigramCollocationFinder.from_words(words)

    # only bi-grams that appear n+ times
    bigram_finder.apply_freq_filter(freq_filter)

    # TODO: use custom stop words
    bigram_finder.apply_word_filter(lambda w: len(w) < 3 or w.lower() in nltk.corpus.stopwords.words('english'))

    bigram_results = bigram_finder.nbest(bigram_measures.likelihood_ratio, nbest)

    return bigram_finder.score_ngrams(bigram_measures.likelihood_ratio)
for r in df['Reviews']:
    if 'sound isolation' in r:
        #print("------")
        #print(r)
        pass
df['Reviews'] = df['Reviews'].apply(lambda x : cleaned_reviews(str(x)))
review_list = df['Reviews'].tolist()
print(len(review_list))
raw_text = ''.join(review_list)
bigrams = get_bigram_likelihood(raw_text.lower(), freq_filter=100, nbest=200 )
bigram_joint = [' '.join(list(s[0])) for s in bigrams]
def check_collocation_type(doc, collocation):
    index = []
    length =len(doc)
    for i in range(length-1):
        if collocation.split(' ')[0] in [doc[i].text, doc[i].lemma_]:
            if collocation.split(' ')[1] in [doc[i+1].text, doc[i+1].lemma_]:
                index.append(i)
    if len(index)> 0:
        return [(doc[index[0]].text, doc[index[0]+1].text),(doc[index[0]].pos_, doc[index[0]+1].pos_)]
    else:
        return False

def select_sentence(review, collocation):
    sentence_list = review.split('.')
    selected_sentences = [sentence for sentence in sentence_list if collocation in sentence]
    
    return ' '.join(selected_sentences)
    
def gen(n):
    while True:
        yield n
        n += 1
selected_collocations_with_type = []
for collocation in bigram_joint:

    selected_data = [select_sentence(review.lower(), collocation) for review in review_list if collocation in review]

    
    g = gen(0)
    while True:
        index = next(g)
        if index >= len(selected_data):
            break
        review = selected_data[index]
        doc= nlp(review)
        check = check_collocation_type(doc, collocation)
        if check != False:
            if  'DET' in check[1]:
                break
            if  'NUM' in check[1]:
                break
            if check[1][1] == 'ADJ':
                break
            if check[1][0] == 'ADJ' and check[1][1] == 'ADV':
                break

            if check[1][0] == 'ADV' and check[1][1] == 'ADV':
                break
            if check[1][1] == 'ADP':
                break
            if check[1][0] == 'ADV' and check[1][1] == 'VERB':
                break
            if check[1][0] == 'ADJ' and check[1][1] == 'VERB':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'ADV':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'NOUN':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'VERB':
                break
            if check[0][1] == 'headphones':
                break
            if check[0][1] == 'product':
                break
            if check[0][0] in ['good','great','amazing','old','first','nice','better','headphones',
                               'long', 'little', 'high', 'low', 'mid','build','price','comfortable',
                              'price','light','second','best','new','reasonable']:
                break
            if check[0][1] in ['loves','phone','time','phones','calls','lasts']:
                break                      

            selected_collocations_with_type.append(check)

            full_data_doc = [(token, token.ent_type_, token.pos_, token.lemma_) for token in doc]
            break
        else:   
            continue



collocation_type_dict = {}
for collocation in selected_collocations_with_type:
    if collocation[1] not in collocation_type_dict:
        collocation_type_dict.update({collocation[1]:[]})
    
    collocation_type_dict[collocation[1]].append(collocation[0])
    
for c in collocation_type_dict:
    print(c," : \n", collocation_type_dict[c])
    print('\n')


selected_collocations = [x[0] for x in selected_collocations_with_type]
bigrams_updated = [x for x in bigrams if x[0] in selected_collocations]
for b in bigrams_updated:
    print(b)



"""
Continuing Preprocessing

"""
"""
Here we replace some of the synonym technical words with the most frequent one
"""
def resolve_heterogeneity(review_list, listOfWords):
    frequency_dict = {w:raw_text.count(w) for w in listOfWords}
    mostFrequency = max(frequency_dict.values())
    selectedWord = [w for w in frequency_dict.keys() if frequency_dict[w]== mostFrequency][0]
    
    for w in listOfWords:
        for i in range(len(review_list)):
            if w in review_list[i]:
                review_list[i] = review_list[i].replace(w, selectedWord)
                
def forced_resolve_heterogeneity(review_list, listOfWords, selectedWord):
    for w in listOfWords:
        for i in range(len(review_list)):
            if w in review_list[i]:
                review_list[i] = review_list[i].replace(w, selectedWord)
                
#Example Noise Canceling, Noise Cancelling, Noise Cancellation, Sound Canceling, Sound Cancelling, Sound Cancellation
resolve_heterogeneity(review_list, ['canceling','cancelling','cancellation','cancelation'])       
raw_text = ''.join(review_list)     
    

# audio, sound
forced_resolve_heterogeneity(review_list, ['audio'], 'sound')
forced_resolve_heterogeneity(review_list, ['bud '], 'buds ')
#noise cancelation, isolation
resolve_heterogeneity(review_list, ['canceling','cancelling','cancellation','cancelation'])
forced_resolve_heterogeneity(review_list, ['outside noise'], 'noise isolation')
forced_resolve_heterogeneity(review_list, ['background noise'], 'noise cancelling')
resolve_heterogeneity(review_list, ['isolating','isolation']) 
resolve_heterogeneity(review_list, ['sound cancelling','noise cancelling']) 
resolve_heterogeneity(review_list, ['sound isolation','noise isolation']) 
#dissimilarity
resolve_heterogeneity(review_list, ['ear','side'])    
resolve_heterogeneity(review_list, ['left','right'])  

resolve_heterogeneity(review_list, ['ear phones','ear cups']) 

forced_resolve_heterogeneity(review_list, ['bluetooth'], 'blue tooth')

#volume control
forced_resolve_heterogeneity(review_list, ['ear monitors'], 'volume control')

# sound quality, quality sound
resolve_heterogeneity(review_list, ['sound quality','quality sound']) 
# ear piece, ear pieces
resolve_heterogeneity(review_list, ['ear piece ','ear pieces ']) 

raw_text = ' '.join(review_list)   
bigrams = get_bigram_likelihood(raw_text.lower(), freq_filter=100, nbest=200 )
bigram_joint = [' '.join(list(s[0])) for s in bigrams]
selected_collocations_with_type = []
for collocation in bigram_joint:

    selected_data = [select_sentence(review.lower(), collocation) for review in review_list if collocation in review]

    
    g = gen(0)
    while True:
        index = next(g)
        if index >= len(selected_data):
            break
        review = selected_data[index]
        doc= nlp(review)
        check = check_collocation_type(doc, collocation)
        if check != False:
            if  'DET' in check[1]:
                break
            if  'NUM' in check[1]:
                break
            if check[1][1] == 'ADJ':
                break
            if check[1][0] == 'ADJ' and check[1][1] == 'ADV':
                break

            if check[1][0] == 'ADV' and check[1][1] == 'ADV':
                break
            if check[1][1] == 'ADP':
                break
            if check[1][0] == 'ADV' and check[1][1] == 'VERB':
                break
            if check[1][0] == 'ADJ' and check[1][1] == 'VERB':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'ADV':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'NOUN':
                break
            if check[1][0] == 'VERB' and check[1][1] == 'VERB':
                break
            if check[0][1] == 'headphones':
                break
            if check[0][1] == 'product':
                break
            if check[0][0] in ['good','great','amazing','old','first','nice','better','headphones',
                               'long', 'little', 'high', 'low', 'mid','build','price','comfortable',
                              'price','light','second','best','new','reasonable']:
                break
            if check[0][1] in ['loves','phone','time','phones','calls','lasts']:
                break            

            selected_collocations_with_type.append(check)

            full_data_doc = [(token, token.ent_type_, token.pos_, token.lemma_) for token in doc]
            break
        else:   
            continue




collocation_type_dict = {}
for collocation in selected_collocations_with_type:
    if collocation[1] not in collocation_type_dict:
        collocation_type_dict.update({collocation[1]:[]})
    
    collocation_type_dict[collocation[1]].append(collocation[0])
    
for c in collocation_type_dict:
    print(c," : \n", collocation_type_dict[c])
    print('\n')




selected_collocations = [x[0] for x in selected_collocations_with_type]
bigrams_updated = [x for x in bigrams if x[0] in selected_collocations]
for b in bigrams_updated:
    print(b)


selected_collocations_joint = [' '.join(bigram) for bigram in selected_collocations]
print(selected_collocations_joint)
"""
Second Step: Sentiment Analysis
"""
"""
second pair
background noise
new pair
"""
selected_collocations
def filter_reviews(review, bigram):
    if bigram in review:
        return True
    else:
        return False
def review_sentiment(string):
    sent = analyser.polarity_scores(string)
    return sent



BrandsList = [ 'Akg', 'BoseAudio', 'Otium', 'Ailhen',  'Shure']
#'Panasonic', 'Sennheiser', , 'sony'
Final_result = {}
Final_dict = {}
for Brand in BrandsList:
    Temp_list = []
    Temp_dict = {}
    brand_df = pd.read_csv('../input/brands/{}_headphones.csv'.format(Brand))
    print(len(brand_df.index))
    brand_df = brand_df.rename(index=str, columns={"Unnamed: 0": "ID"})
    brand_df['Reviews'] = brand_df['Reviews'].apply(lambda x : cleaned_reviews(str(x)))
    brand_reviews = brand_df['Reviews'].tolist()
    brand_reviews = [str(x) for x in brand_reviews]
    brand_raw_text = ' '.join(brand_reviews)
    brand_bigrams =get_bigram_likelihood(brand_raw_text, freq_filter=3, nbest=200 )
    
    for bigram in selected_collocations_joint:
        Filtered_list = [select_sentence(review, bigram) for review in brand_reviews if filter_reviews(review, bigram)]

        if len(Filtered_list) > 0:
            df = pd.DataFrame({"Reviews":Filtered_list})
            df['Sentiments'] = df['Reviews'].apply(lambda x: review_sentiment(x))
            df = pd.concat([df.drop(['Sentiments'], axis=1), df['Sentiments'].apply(pd.Series)], axis=1)
            negativity = df['neg'].sum()
            #print(negativity)
            positivity = df['pos'].sum()
            Temp_list.append((bigram,positivity/(positivity+negativity)))
            Temp_dict.update({bigram:positivity/(positivity+negativity)})
        else:
            Temp_dict.update({bigram:-1})
            
    Final_result.update({Brand: sorted(Temp_list, key = lambda x:x[1])}) 
    Final_dict.update({Brand: Temp_dict})    
    print("Brand : ", Brand)
    print("Bad Features:")
    for b in Final_result[Brand][:6]:
        print(b)
    print("Good Features:")
    for b in Final_result[Brand][-6:]:
        print(b)
    print('\n')

    



Final_table = np.array([[Final_dict[Brand][collocation] for collocation in 
                         selected_collocations_joint] for Brand in BrandsList])
numOfBrands = len(Final_table)
numOfCollocations = len(Final_table[0])


def regularize(myTable):
    n = len(myTable)
    m = len(myTable[0])
    min_table = min([min(myTable[j]) for j in range(n)])
    max_table = max([max(myTable[j]) for j in range(n)])
    return [[(myTable[j][i]-min_table)/(max_table - min_table)  for i in range(m) ] 
               for j in range(n)]


"""
We have 3 missed data for Bose and AKG brands: Bluetooth, Battery life, Foam tips
"""
Final_table = [[Final_table[j][i] if Final_table[j][i]>-1 
                else np.mean([Final_table[k][i] for k in range(numOfBrands) if Final_table[k][i]>-1])
                             for i in range(numOfCollocations) ] 
               for j in range(numOfBrands)]

Final_table = np.array(regularize(Final_table))
#
