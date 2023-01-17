# Load libraries

import pandas as pd
import numpy as np 
import string
import random

import nltk
from nltk.corpus import brown
from nltk.corpus import reuters

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
#load 10k reuters news documents 
len(reuters.fileids())
#view text from one document 
reuters.raw(fileids=['test/14826'])[0:203]
# remove punctuation from all DOCs 
exclude = set(string.punctuation)
alldocslist = []

for index, i in  enumerate(reuters.fileids()):
    text = reuters.raw(fileids=[i])
    text = ''.join(ch for ch in text if ch not in exclude)
    alldocslist.append(text)
    
print(alldocslist[1])
#tokenize words in all DOCS 
plot_data = [[]] * len(alldocslist)

for doc in alldocslist:
    text = doc
    tokentext = word_tokenize(text)
    plot_data[index].append(tokentext)
    
print(plot_data[0][1])
# Navigation: first index gives all documents, second index gives specific document, third index gives words of that doc
plot_data[0][1][0:10]
# remove stop words from all docs 
stop_words = set(stopwords.words('english'))

for x in range(len(reuters.fileids())):
    filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
    plot_data[0][x] = filtered_sentence

plot_data[0][1][0:10]
#stem words EXAMPLE (could try others/lemmers )

snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
stemmed_sentence[0:10]

porter_stemmer = PorterStemmer()
snowball_stemmer = SnowballStemmer("english")
stemmed_sentence = [ porter_stemmer.stem(w) for w in filtered_sentence]
stemmed_sentence[0:10]
# Create inverse index which gives document number for each document and where word appears

#first we need to create a list of all words 
l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)
# create functions for TD-IDF / BM25
import math
from textblob import TextBlob as tb

def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))

def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))
# Create dictonary of words
# THIS ONE-TIME INDEXING IS THE MOST PROCESSOR-INTENSIVE STEP AND WILL TAKE TIME TO RUN (BUT ONLY NEEDS TO BE RUN ONCE)
import re
import numpy as np

plottest = plot_data[0][0:1000]

worddic = {}

for doc in plottest:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = plottest.index(doc)
            positions = list(np.where(np.array(plottest[index]) == word)[0])
            idfs = tfidf(word,doc,plottest)
            try:
                worddic[word].append([index,positions,idfs])
            except:
                worddic[word] = []
                worddic[word].append([index,positions,idfs])
# the index creates a dic with each word as a KEY and a list of doc indexs, word positions, and td-idf score as VALUES
worddic['china']
# pickel (save) the dictonary to avoid re-calculating
np.save('worddic_1000.npy', worddic)
# create word search which takes multiple words and finds documents that contain both along with metrics for ranking:

    ## (1) Number of occruances of search words 
    ## (2) TD-IDF score for search words 
    ## (3) Percentage of search terms
    ## (4) Word ordering score 
    ## (5) Exact match bonus 


from collections import Counter

def search(searchsentence):
    try:
        # split sentence into individual words 
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
        closedic = {}
        
        # remove words if not in worddic 
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)  
        words = realwords
        numwords = len(words)
        
        # make metric of number of occurances of all words in each doc & largest total IDF 
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)

                
        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], reverse=True)
        
        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                     x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y))

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions)

            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol = [1 for i in firstlist if i + 1 in secondlist]
                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x:x[1], reverse=True)
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = 0
                    
        # also the one above should be given a big boost if ALL found together 
           
        
        #could make another metric for if they are not next to each other but still close 
        
        
        return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
    
    except:
        return("")


search('indonesia crude palm oil')[1]
search('indonesia crude palm oil')[1][1:10]
# save metrics to dataframe for use in ranking and machine learning 
result1 = search('crude')
result2 = search('indonesia crude palm oil')
result3 = search('price of nickel')
result4 = search('north yemen sugar')
result5 = search('nippon steel')
result6 = search('sugar')
result7 = search('Gold')
result8 = search('trade')
df = pd.DataFrame([result1,result2,result3,result4,result5,result6,result7,result8])
df.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order']

print(result1)
# look to see if the top documents seem to make sense
alldocslist[1]
# create a simple (non-machine learning) rank and return function

def rank(term):
    results = search(term)

    # get metrics 
    num_score = results[2]
    per_score = results[3]
    tfscore = results[4]
    order_score = results[5]
    
    final_candidates = []

    # rule1: if high word order score & 100% percentage terms then put at top position
    try:
        first_candidates = []

        for candidates in order_score:
            if candidates[1] > 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])

    # rule2: next add other word order score which are greater than 1 

        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule3: next add top td-idf results
        final_candidates.insert(len(final_candidates),tfscore[0][0])
        final_candidates.insert(len(final_candidates),tfscore[1][0])

    # rule4: next add other high percentage score 
        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    #rule5: next add any other top results for metrics
        othertops = [num_score[0][0],per_score[0][0],tfscore[0][0],order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)
                
    # unless single term searched, in which case just return 
    except:
        othertops = [num_score[0][0],num_score[1][0],num_score[2][0],per_score[0][0],tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)

    for index, results in enumerate(final_candidates):
        if index < 5:
            print("RESULT", index + 1, ":", alldocslist[results][0:100],"...")
# example of output 
rank('indonesia palm oil')
# example of output 
rank('palm oil')
# Create pseudo-truth set using first 5 words 
# Because I don't have a turth set I will generate a pseudo one by pulling terms from the documents - this is far from perfect
     # as it may not approximate well peoples actual queries but it will serve well to build the ML architecture 

df_truth = pd.DataFrame()

for doc in plottest:
    first_five = doc[0:5]
    test_sentence = ' '.join(first_five)
    result = search(test_sentence)
    df_temp = pd.DataFrame([result])
    df_truth= pd.concat([df_truth, df_temp])

df_truth['truth'] = range(0,len(plottest))
# create another psuedo-truth set using random 3 word sequence from docs

df_truth1 = pd.DataFrame()
seqlen = 3

for doc in plottest:
    try:
        start = random.randint(0,(len(doc)-seqlen))
        random_seq = doc[start:start+seqlen]
        test_sentence = ' '.join(random_seq)
    except:
        test_sentence = doc[0]
    result = search(test_sentence)
    df_temp = pd.DataFrame([result])
    df_truth1= pd.concat([df_truth1, df_temp])

df_truth1['truth'] = range(0,len(plottest))
# create another psuedo-truth set using different random 4 word sequence from docs

df_truth2 = pd.DataFrame()
seqlen = 4

for doc in plottest:
    try:
        start = random.randint(0,(len(doc)-seqlen))
        random_seq = doc[start:start+seqlen]
        test_sentence = ' '.join(random_seq)
    except:
        test_sentence = doc[0]
    result = search(test_sentence)
    df_temp = pd.DataFrame([result])
    df_truth2= pd.concat([df_truth2, df_temp])

df_truth2['truth'] = range(0,len(plottest))
# create another psuedo-truth set using different random 2 word sequence from docs

df_truth3 = pd.DataFrame()
seqlen = 2

for doc in plottest:
    try:
        start = random.randint(0,(len(doc)-seqlen))
        random_seq = doc[start:start+seqlen]
        test_sentence = ' '.join(random_seq)
    except:
        test_sentence = doc[0]
    result = search(test_sentence)
    df_temp = pd.DataFrame([result])
    df_truth3= pd.concat([df_truth3, df_temp])

df_truth3['truth'] = range(0,len(plottest))
# combine the truth sets and save to disk 
truth_set = pd.concat([df_truth,df_truth1,df_truth2,df_truth3])
truth_set.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order','truth']
truth_set.to_csv("truth_set_final.csv")
truth_set[0:10]
truth_set
test_set = truth_set[0:3]
test_set
# convert to long format for ML 
# WARNING AGAIN THIS IS A SLOW PROCESS DUE TO RAM ILOC - COULD BE OPTIMISED FOR FASTER PERFORMANCE 
# BUG When min(maxnum, len(truth_set) <- is a int not a list because of very short variable length)

# row is row
# column is variable
# i is the result 
import math
import numpy as np
final_set =  pd.DataFrame()
test_set = truth_set[1:100]
maxnum = 5

for row in range(0,len(test_set.index)):
    test_set = truth_set[1:100]
    for col in range(2,6):
        print(truth_set.iloc[row][col])
        
        if not np.any(np.isnan(truth_set.iloc[row][col]) ): 
            if truth_set.iloc[row][col] :
                for i in range(0,min(maxnum,len(truth_set.iloc[row][col]))):
                    x = pd.DataFrame([truth_set.iloc[row][col][i]])
                    x['truth'] = truth_set.iloc[row]['truth']
                    x.columns = [(str(truth_set.columns[col]),"index",i),(str(truth_set.columns[col]),"score",i),'truth']
                    test_set = test_set.merge(x,on='truth')
                final_set = pd.concat([final_set,test_set])
        
final_set.head()
final_set.to_csv("ML_set_100.csv")
final_set2 = final_set.drop(['actual_words_searched','num_occur','percentage_of_terms','search term','td-idf','word_order'], 1)
final_set2.to_csv("ML_set_100_3.csv")
final_set2.head()
final_set3 = final_set2
final_set3[0:10]
# Load libraries 
%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as sm
import statsmodels.api as sma
from statsmodels.tools.eval_measures import mse
from statsmodels.tools.tools import add_constant

from sklearn import linear_model,cross_validation, feature_selection,preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import KFold
final_set3['y'] = final_set3['truth']
final_set3 = final_set3.drop(['truth'], 1)
final_set3
data = final_set3
data.corr()['y']
data['a'] = data[data.columns[0]]
data['b'] = data[data.columns[10]]
data['c'] = data[data.columns[20]]
data['d'] = data[data.columns[30]]
X = data

train,test = cross_validation.train_test_split(X,train_size=0.80)

model = sm.ols(formula='y ~ 1 + a + b + c + d', 
               data=train).fit()

modelforout = model 

model.summary()
fig, ax = plt.subplots(figsize=(12,8))
fig = sma.graphics.influence_plot(modelforout, ax=ax, criterion="cooks")
res = model.resid # residuals
fig = sma.qqplot(res)
plt.show()
term = input("search: ")
rank(term)
