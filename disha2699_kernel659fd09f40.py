import pandas as pd
import numpy as np 
import nltk
import re
from wordcloud import WordCloud ,STOPWORDS
import spacy

import textblob
from textblob import Word
from textblob import TextBlob
import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
nlp = spacy.load('en_core_web_sm')
stop = set(nltk.corpus.stopwords.words('english'))
import string
def pre_process(text):
    #text = " ".join(word.lower() for word in text.split())
    text = re.sub("  "," ", text)         #Replacing double space with single space
    text = re.sub(r'''[-()\"#/@;:{}`+=~|.!?,']''', "", text)     #Replacing special character with none
    text = " ".join(text for text in text.split() if text not in stop) #Removing stop words
    return(text)
sentence = "Output Revenue, EBITDA margin for Steel and Metal stocks for past 10 qtrs"
sentence = pre_process(sentence)
words = sentence.split()
words
from itertools import permutations  
    
def error_permutator( st): 
    start = st[0]
    st = st[1:] 
    # Number of subsequences is (2**n -1) 
    n = len(st)
    opsize = pow(2, n) 
   
    # Generate all subsequences of a given string. 
    #  using counter 000..1 to 111..1 
    for counter in range(1, opsize): 
      
        subs = "" 
        for j in range(n): 
          
            # Check if jth bit in the counter is set 
            #   If set then print jth element from arr[]  
            if (counter & (1<<j)): 
                subs += (st[j]) 
   
        # Print all permutations of current subsequence  
        perm = permutations(subs) 
          
        for i in perm: 
            errors[start+''.join(i)] = start+st
errors = {}
time_period = ['years', 'year', 'quarter', 'quarters', 'month', 'months', 'day', 'days', 'hours', 'hour']
for val in time_period:
    error_permutator(val)
properties = dict.fromkeys(['Sector' , 'Fundamentals', 'Time Period' , 'Time'])
properties
def no_from_string(word):
    try:
        float(word)
        return True
    except:
        return False 
for i in range(len(words)):
    if no_from_string(words[i]):
        if words[i+1] in errors.keys():
            print('Time:', int(words[i]))
            words[i+1] = errors[words[i+1]]
            print('Time Period:', words[i+1])

words = [str(TextBlob(x).correct()) for x in words]
Sectors = ['Cement', 'Fertilizers', 'Trading', 'Pharmaceuticals', 'Paper', 'Bearings', 'Tyres', 'Textiles', 'Automobile', 'Hotels &                  Restaurants', 'Paints/Varnish', 'Mining & Mineral products', 'Chemicals', 'Auto Ancillaries', 'Finance', 'Consumer Durables',              'Sugar','Leather', 'Agro Chemicals', 'Capital Goods - Electrical Equipment', 'Castings, Forgings & Fastners', 'Plantation &                Plantation Products', 'Power Generation & Distribution', 'Glass & Glass Products', 'FMCG', 'Capital Goods-Non Electrical                Equipment','Construction', 'Packaging', 'Petrochemicals', 'Cement – Products', 'Cables', 'Miscellaneous', 'Engineering',                 'Entertainment','Shipping','Sanitaryware', 'Non Ferrous Metals', 'IT – Software', 'Steel', 'Tobacco Products', 'Alcoholic                  Beverages','Diversified', "Dry cells", 'Retail', 'Infrastructure Developers & Operators', 'Realty', 'Telecomm Equipment &                 Infra Services', 'Refineries', 'Ceramic Products','Plastic products', 'Healthcare', 'Edible Oil', 'Diamond, Gems and Jewellery', 'IT - Hardware', 'Refractories', 'Crude Oil & Natural Gas','Printing & Stationery', 'Oil Drill/Allied', 'Electronics', 'Banks', 'Telecom-Handsets/Mobile', 'Logistics', 'Media - Print/Television/Radio', 'Education', 'Telecomm-Service', 'Readymade Garments/ Apparells', 'Computer Education', 'Credit Rating Agencies', 'Air Transport Service', 'Stock/ Commodity Brokers', 'Insurance', 'Ship Building', 'Marine Port & Services', 'Gas Distribution', 'Infrastructure Investment Trusts','Power Infrastructure']
fundamentals = ['Revenue', 'Inventory']  # This list will further expand when a proper fundamentals list is                                                                        # there
## patching Syntatic similarities in Sectors
for word in Sectors:
    for dat in words:
        if word in dat:
            properties['Sector'].append(word)
sen = " ".join(word for word in words)
if re.search('Revenue',sen):
    sen_tokenized = nltk.word_tokenize(sen)
    tagged = nltk.pos_tag(sen_tokenized)
tagged
properties = dict.fromkeys(['Sector' , 'Fundamentals', 'Time Period' , 'Time'])
properties
sentence = "Output Revenue, EBITDA margin for Steel and Metal stocks for past 10 qtrs".split('for')
sentence
time_parser_list = []
sector_parser_list = []
fundamental_parser_list = []
number_in_sen = False
period_in_sen = False

# Extracting the list with time
for sen in sentence:
    sen_token = nltk.word_tokenize(sen)
    for word in sen_token:
        if no_from_string(word):
            number_in_sen = True
        if word in errors.keys():
            period_in_sen = True
    if( number_in_sen and period_in_sen):
        time_parser_list = sen_token

## Extracting sub sentence with sectors and fundamentals
for sen in sentence:
    sen_token = nltk.word_tokenize(sen)
    for word in sen_token:
        if word in Sectors:
            sector_parser_list = sen_token
        elif word in fundamentals:
            fundamental_parser_list = sen_token
time_parser_list
sector_parser_list
fundamental_parser_list
time_sen = ' '.join(word for word in time_parser_list)
for i in range(len(time_parser_list)):
    if no_from_string(time_parser_list[i]):
        if time_parser_list[i+1] in errors.keys():
            properties['Time'] = int(time_parser_list[i])
            time_parser_list[i+1] = errors[time_parser_list[i+1]]
            properties['Time Period'] = time_parser_list[i+1]
properties
# Cleaning
sector_sen = " ".join(word for word in sector_parser_list)
sector_sen = pre_process(sector_sen) # will remove all stop words, useless signs 
sen_tokenized = nltk.word_tokenize(sector_sen)
tagged = nltk.pos_tag(sen_tokenized)
tagged
properties['Sector'] = []
for tags in tagged:
    if 'NNP' in tags or 'NN' in tags or tags[0] in Sectors:
        properties['Sector'].append(tags[0]) 


properties['Sector']
# Cleaning
Fundamentals_sen = " ".join(word for word in fundamental_parser_list)
Fundamentals_sen = pre_process(Fundamentals_sen) # will remove all stop words, useless signs 
sen_tokenized = nltk.word_tokenize(Fundamentals_sen)
tagged = nltk.pos_tag(sen_tokenized)
tagged
properties['Fundamentals'] = []
fundamental_parser_list
for i in range(len(fundamental_parser_list)):
    if fundamental_parser_list[i].isupper():
        if not fundamental_parser_list[i+1].isupper():
            properties['Fundamentals'].append(fundamental_parser_list[i]+' '+fundamental_parser_list[i+1])
        else:
            properties['Fundamentals'].append(fundamental_parser_list[i])
for tags in tagged:
    if ('NNP' in tags or 'NN' in tags) and tags[0] in fundamentals:
        properties['Fundamentals'].append(tags[0])  

properties['Fundamentals']
properties
