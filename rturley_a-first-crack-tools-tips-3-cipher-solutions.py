# if using from the AISE.ai machine image on Google Cloud Platform,
# upload the kaggle.json file to /jet/prs/workspace, and then from SSH:
# cd /home/jet
# mkdir .kaggle
# cd /jet/prs/workspace
# cp kaggle.json /home/jet/.kaggle
#  chmod 600 /home/jet/.kaggle/kaggle.json
newdownloads = False
if newdownloads:
    # install missing packages
    !pip install matplotlib
    !pip install scikit-learn
    !pip install fuzzywuzzy
    !pip install python-Levenshtein
    !pip install tqdm
    !pip install kaggle
    !pip install "dask[complete]"
    !pip install graphviz
    
    # download the kaggle data
    import os
    import zipfile
    DATA_DIR = '/jet/prs/workspace/data'
    os.makedirs(DATA_DIR)
    os.chdir(DATA_DIR)
    !kaggle competitions download -c 20-newsgroups-ciphertext-challenge
    #unzip
    trainzip = zipfile.ZipFile('train.csv.zip')
    trainzip.extractall(path=DATA_DIR)
    testzip = zipfile.ZipFile('test.csv.zip')
    testzip.extractall(path=DATA_DIR)
import os

import numpy as np
np.random.seed(seed=42) # fix seed for reproduceability
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.feature_extraction.text import CountVectorizer
token_pattern = r"[a-zA-Z0-9!@#$&()<>'=\-]+" # CountVectorizer word token

from fuzzywuzzy import fuzz, process
import Levenshtein as leven

from itertools import compress
from tqdm import tqdm

# distributed computing
from dask import delayed, compute
from dask.diagnostics import ProgressBar
ProgressBar().register()
# Kaggle test and training data
onkaggle = True
if onkaggle:
    train_datafile = '../input/train.csv'
    test_datafile = '../input/test.csv'
else:
    train_datafile = '/jet/prs/workspace/data/train.csv'
    test_datafile = '/jet/prs/workspace/data/test.csv'
train_df = pd.read_csv(train_datafile)
test_df =  pd.read_csv(test_datafile)
test_df['length'] = np.array([len(test_df['ciphertext'].iloc[idx]) for idx in range(len(test_df))],dtype=int)
print("Train data")
train_df.tail(5)
print("Test data")
test_df.tail(5)
%%time
from sklearn.datasets import fetch_20newsgroups
twenty_databunch = fetch_20newsgroups(subset='all', download_if_missing=True)

# CORRECTION TO MATCH KAGGLE DATA
def sourcetransform(textstring):
    return textstring.replace('\r\n','\n').replace('\r','\n').replace('\n','\n ').rstrip(' ')

sourcetext = twenty_databunch.data
for i,textstring in enumerate(sourcetext):
    sourcetext[i] = sourcetransform(textstring)
    
twenty_databunch.data = sourcetext

twenty_datalengths = [len(datastring) for datastring in twenty_databunch.data]

category_names = twenty_databunch.target_names


chunks_plaintext = []
chunks_target = []
chunks_length = []
for i in range(len(twenty_databunch.target)):
    strlength = len(twenty_databunch.data[i])
    if strlength > 300:
        for j in range(strlength // 300):
            chunks_plaintext.append(twenty_databunch.data[i][300*j:300*(j+1)])
            chunks_target.append(twenty_databunch.target[i])
            chunks_length.append(300)
        if strlength%300 > 0:
            chunks_plaintext.append(twenty_databunch.data[i][300*(strlength // 300):(300*(strlength // 300)+strlength%300)])
            chunks_target.append(twenty_databunch.target[i])
            chunks_length.append(strlength%300)
    else:
        chunks_plaintext.append(twenty_databunch.data[i])
        chunks_target.append(twenty_databunch.target[i])
        chunks_length.append(strlength)
        
chunk_df = pd.DataFrame({'plaintext':chunks_plaintext,
                         'length':np.array(chunks_length,dtype=int), 
                         'target':np.array(chunks_target,dtype=int)})
chunk_df['testref'] = np.nan
chunk_df['trainref'] = np.nan


# is is very helpful to have a copy of the data in dictionary form for fuzzy wuzzy lookup
sourcetext_dict = {idx: el for idx, el in enumerate(chunks_plaintext)}
chunk_df.tail()
# translating a string of text to an array of 8-bit integers representing ASCII values
def string2ascii(textstring):
    return np.array([ord(char) for char in textstring], dtype=np.int8)

# translating an array of 8-bit integers representing ASCII values to a string
def ascii2string(nparray):
    return ''.join(chr(npint) for npint in nparray)

# calculate ascii character frequency per million characters
def char_per_million(stringsarray):
    asciicount = np.zeros((128,), dtype=int)
    for k,textdata in enumerate(stringsarray):
        asciicount += np.histogram(string2ascii(textdata),np.arange(129))[0]
    totalchars = np.sum(asciicount)
    return np.multiply(asciicount,np.divide(1000000.0,totalchars))

# substitution decipher using the input asciimap dataframe (index= ciphered ascii integer)
def decipher_subst(textstring,asciimap):
    inarray = string2ascii(textstring)
    outarray = np.zeros(len(inarray),dtype=np.int8)
    for asciival in asciimap.index:
        outarray[inarray==asciival] = asciimap['decipher'].loc[asciival]
    return ascii2string(outarray)

# For fine-tuning a cipher, this identifies letter character replacements needed
# The output is a 128x128 matrix with rows and columns corresponding to ascii characters 
# The element in element [a,b] is the number of times the ascii character in texta had to 
# be switched for the ascii character in textb
def leven_replace_matrix(textA,textB):
    # initialize output matrix
    asciiswitchmat = np.zeros([128,128],dtype=int)
    # calculate edits for Levenshtein distance
    lops = leven.editops(textA,textB)
    # count the character replacements needed
    replacetf = [editop[0] is 'replace' for editop in lops]
    replacerefA =  np.array([editop[1] for editop in lops],dtype=np.int8)
    replacerefB =  np.array([editop[2] for editop in lops],dtype=np.int8)
    goodidx = np.logical_and(replacetf,np.logical_and(replacerefA<128,replacerefB<128))
    replacerefA = replacerefA[replacetf]
    replacerefB = replacerefB[replacetf]
    for i, refA in enumerate(replacerefA):
        asciiswitchmat[ord(textA[refA]),ord(textB[replacerefB[i]])] += 1
    return asciiswitchmat
def char_count(textstring):
    return np.histogram(string2ascii(textstring),np.arange(129))[0]
    
# calculate ascii character frequency per million characters
def char_per_300(stringsarray):
    asciicount = np.zeros((128,), dtype=int)
    for k,textdata in enumerate(stringsarray):
        asciicount += char_count(textdata)
    totalchars = np.sum(asciicount)
    return np.multiply(asciicount,np.divide(300.0,totalchars))

# number of words
def num_recognized_words(stringsarray,worddictionary):
    vectorizer = CountVectorizer(analyzer='word',vocabulary=worddictionary2)
    X = vectorizer.fit_transform(stringsarray)
    return np.sum(X)

# calculate words per million characters using a given dictionary list of ngrams
# the worddictionary is generated on a previous run of CountVectorizer using English text
def word_per_million(stringsarray,worddictionary):
    totalchars = sum([len(cipherstring) for cipherstring in stringsarray])
    vectorizer = CountVectorizer(analyzer='word',token_pattern=token_pattern,vocabulary=worddictionary)
    #vectorizer = CountVectorizer(vocabulary=worddictionary)
    X = vectorizer.fit_transform(stringsarray)
    return np.squeeze(np.asarray(np.sum(X,axis=0)*(1000000.0/totalchars)))

# calculate ngrams per million characters using a given dictionary list of ngrams
# the ngramdictionary is generated on a previous run of CountVectorizer using English text
def ngram_per_million(stringsarray,ngramdictionary):
    ngramlength = len(next(iter(ngramdictionary)))
    totalchars = sum([len(cipherstring) for cipherstring in stringsarray])
    vectorizer = CountVectorizer(analyzer='char', 
                             max_features=1000,
                             lowercase = False,
                             ngram_range = (n_ngram,n_ngram),
                             vocabulary=ngramdictionary)
    X = vectorizer.fit_transform(stringsarray)
    return np.squeeze(np.asarray(np.sum(X,axis=0)*(1000000.0/totalchars)))

def similarity_score(x, y):
    return 1.0 - np.divide(np.linalg.norm(x - y),np.linalg.norm(y))

# goodnes of fit
def goodness_of_fit(cipher_array,asciimap,verbose=True):
    decipher_array = [decipher_subst(ciphertext,asciimap) 
     for ciphertext in cipher_array]
    w_score = num_recognized_words(decipher_array, worddictionary)
    wpm_candidate = word_per_million( decipher_array, worddictionary)
    npm_candidate = ngram_per_million( decipher_array, ngramdictionary)
    wpm_score = similarity_score(wpm_candidate, wpm_plaintext)
    npm_score = similarity_score(npm_candidate, npm_plaintext)
    if verbose:
        print('   Goodness-of-Fit vs Source Text   ')
        print('====================================')
        print("Number of Recognized Words = " + str(w_score))
        print("Word-per-Million Similarity = " + str(wpm_score))
        print("Ngram-per-Million Similarity = " + str(npm_score))
    return w_score,wpm_score, npm_score
# Display an example data item along with its associated category and filename
dataitem = 0
print(twenty_databunch.data[dataitem])
fig, ax = plt.subplots(figsize=(9, 12))
ax.barh(np.arange(len(category_names)),
        [sum(twenty_databunch.target==k) for k in range(len(category_names))],
        tick_label = category_names)
ax.tick_params(axis='both',labelsize=14)
plt.title('# of Newsgroup Postings',fontsize=16)
plt.show()
print(str((100*sum(chunk_df['length']==300))//len(chunk_df))+'% of the data items have a size = 300')
print(str(sum(chunk_df['length']<5))+' of the data items have a size < 5 characters!')
%%time
cpm_plaintext = char_per_million(chunks_plaintext)
asciitop = np.argsort(-1*cpm_plaintext)
asciitop_df = pd.DataFrame({'char':[chr(asciinum) for asciinum in asciitop],'char_per_million':cpm_plaintext[asciitop]},index=asciitop)
# create a bar chart to highlight the top 40 most common characters
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(np.arange(30),asciitop_df['char_per_million'].iloc[0:30],color='red', marker='*', markersize=10)
ax.set_xticklabels(["'"+chr(asciinum)+"'" for asciinum in asciitop_df.index[0:30]])
ax.set_xticks(np.arange(30))
ax.tick_params(axis='both',labelsize=16)
plt.ylabel('Occurrence per Million Characters',fontsize=16)
plt.title('Top 30 ASCII Characters: Plaintext Source',fontsize=18)
plt.show()
print("Of the 128 possible ASCII values, only "+ str(sum(cpm_plaintext>1)) +" are used more than once per million")
# the 5000 most common traditional words
totalchars = sum([len(cipherstring) for cipherstring in twenty_databunch.data])
vectorizer = CountVectorizer(analyzer='word', max_features=5000)
source_words = vectorizer.fit_transform(twenty_databunch.data)
worddictionary2 = vectorizer.vocabulary_

# the 5000 most common words, including one-letter words and symbols
totalchars = sum([len(cipherstring) for cipherstring in twenty_databunch.data])
vectorizer = CountVectorizer(analyzer='word', max_features=5000, token_pattern=token_pattern)
source_words = vectorizer.fit_transform(twenty_databunch.data)
worddictionary = vectorizer.vocabulary_
# the word-per-million score from plaintext English that we try to match in deciphering
wpm_plaintext = word_per_million(twenty_databunch.data,worddictionary)

# the 1000 most common English ngrams
n_ngram = 3
totalchars = sum([len(cipherstring) for cipherstring in twenty_databunch.data])
vectorizer = CountVectorizer(analyzer='char', 
                             max_features=1000,
                             lowercase = False,
                             ngram_range = (n_ngram,n_ngram) )
source_words = vectorizer.fit_transform(twenty_databunch.data)
ngramdictionary = vectorizer.vocabulary_
# the word-per-million score from plaintext English that we try to match in deciphering
npm_plaintext = ngram_per_million(twenty_databunch.data,ngramdictionary)
# this dictionary dataframe will show the common words
worddict_df = pd.DataFrame.from_dict(worddictionary, orient='index')
worddict_df.reset_index(inplace=True)
worddict_df.rename(index=str, columns={"index": "word", 0: "dict_index"},inplace=True)
worddict_df.set_index('dict_index',inplace=True)
worddict_df.sort_index(inplace=True)
worddict_df['words_per_million'] = wpm_plaintext
worddict_df.sort_values('words_per_million',ascending=False,inplace=True)
print("Top 10 Most Common Words/Symbols")
print(worddict_df.head(10))
# we combine test and training data for greatest precision
cpm_train = char_per_million(train_df[train_df['difficulty']==1]['ciphertext'])
cpm_test =  char_per_million(test_df[test_df['difficulty']==1]['ciphertext'])
cpm_cipher = 0.67*cpm_test + 0.33*cpm_train

c1asciitop = np.argsort(-1*cpm_cipher)
c1asciitop_df = pd.DataFrame({'char':[chr(asciinum) for asciinum in c1asciitop],
                            'char_per_million':cpm_cipher[c1asciitop]},index=c1asciitop)
# create a bar chart to highlight the top 40 most common characters
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(np.arange(30),
       c1asciitop_df['char_per_million'].iloc[0:30],
       tick_label = ["'"+chr(asciinum)+"'" for asciinum in c1asciitop_df.index[0:30]] )
ax.plot(np.arange(30),asciitop_df['char_per_million'].iloc[0:30],color='red', marker='*', markersize=10)
ax.tick_params(axis='both',labelsize=16)
plt.ylabel('Occurrence per Million Characters',fontsize=16)
plt.title('Top 30 ASCII Characters: Ciphertext Difficulty=1',fontsize=18)
plt.legend(['Distribution for Top Characters in Source Data','Distribution for Difficulty=1'],fontsize=16)
plt.show()
print("Of the 128 possible ASCII values, only "+ str(sum(cpm_cipher>1)) +" are used more than once per million")
asciimap = pd.DataFrame(asciitop, index=c1asciitop,columns=['decipher'])
textstring = train_df[train_df['difficulty']==1]['ciphertext'].iloc[0]
print('SAMPLE DECIPHER ATTEMPT:')
print(decipher_subst(textstring,asciimap))
%%time
# Go through the plain text from the category and find the matching entry
textstring = decipher_subst(train_df[train_df['difficulty']==1]['ciphertext'].iloc[0],asciimap)
textcategory = train_df[train_df['difficulty']==1]['target'].iloc[0]
sourcematch = process.extractOne(textstring, sourcetext_dict, scorer = fuzz.ratio)
print('Found a match in source data with "fuzz.ratio" score of ' + str(fuzz.ratio(textstring,sourcematch[0])))
print('Source data found in category = ' + category_names[chunk_df['target'].iloc[sourcematch[2]]])
print('Matching text below:')
print('----------------')
print(sourcematch[0])
print(' ')
# subsample of ciphertext
subset_size = 2000
tune_df = train_df[np.logical_and(train_df['difficulty']==1,
                                    np.array([len(train_df['ciphertext'].iloc[idx]) for idx in range(len(train_df))])>100)]
tuneindices = tune_df.index[:subset_size]
        
# function to match to source data for same category and length >100 chars
def finetunematch(idx):
    textstring = decipher_subst(tune_df['ciphertext'].loc[idx],asciimap)
    selectindices = np.logical_and(chunk_df['target']==tune_df['target'].loc[idx],chunk_df['length']>250)
    sourcematch = process.extractOne(textstring, 
                                     list(compress(chunks_plaintext, selectindices)), 
                                     scorer = fuzz.ratio)
    return sourcematch[0], textstring, sourcematch[1]

# parallel evaluation using Dask (big benefits for more CPUs)
par_compute = [delayed(finetunematch)(idx) for idx in tuneindices]
output_arrays = compute(*par_compute, scheduler='processes')
# select those with relatively high fuzzy.ratio scores
minfuzzyscore = 80
sourcematch_array = [arrays[0] for arrays in output_arrays]
decipher_array = [arrays[1] for arrays in output_arrays]
fuzzyscores_array = np.array([arrays[2] for arrays in output_arrays])
sourcematch_array = list(compress(sourcematch_array, (fuzzyscores_array>=minfuzzyscore) ))
decipher_array = list(compress(decipher_array, (fuzzyscores_array>=minfuzzyscore) ))
cipher_array =  tune_df[:subset_size]['ciphertext']
cipher_array = list(compress(cipher_array, (fuzzyscores_array>=minfuzzyscore) ))
print('In subsample, ' + str(np.sum(fuzzyscores_array>=minfuzzyscore)) + ' of the ' + str(subset_size) + ' have fuzzy.ratios > ' 
       + str(minfuzzyscore) + ' and will be used for fine-tuning')
print(' ')
print('Before fine-tuning the sub-sample:')
gof = goodness_of_fit(cipher_array,asciimap=asciimap)
print('Average fuzzy.ratio = ' + str(np.mean(fuzzyscores_array[fuzzyscores_array>=minfuzzyscore])))
# initialize asciiswitchmat, a 128x128 matrix with rows numbers corresponding to ascii characters
# that should be replaced by the ascii character of the column number
asciiswitchmat = np.zeros([128,128],dtype=int)
asciicount = np.zeros((128,), dtype=int)

for i, textstring in enumerate(decipher_array):
    asciiswitchmat += leven_replace_matrix(textstring,sourcematch_array[i])
    asciicount += np.histogram(string2ascii(textstring),np.arange(129))[0]

deciphererror_dict = {'ascii_replacement': [np.argmax(asciiswitchmat[i,...]) for i in range(128)],
                      'char_decipher': [chr(i) for i in range(128)],
                      'char_replacement': [chr(np.argmax(asciiswitchmat[i,...])) for i in range(128)],
                      'numError': np.sum(asciiswitchmat,axis=1),
                      'numReplace': [asciiswitchmat[i,np.argmax(asciiswitchmat[i,...])] for i in range(128)],
                      'totalObs': asciicount}

deciphererror_df = pd.DataFrame.from_dict(deciphererror_dict)
deciphererror_df['pctError'] = deciphererror_df['numError'] / deciphererror_df['totalObs']
deciphererror_df['pctBestReplace'] = deciphererror_df['numReplace'] / deciphererror_df['numError']  
deciphererror_df.sort_values(by=['pctError'],ascending=False,inplace=True)
deciphererror_df.head(10)
# correct the identified errors in character substitution
errorprone = np.logical_and(deciphererror_df['totalObs']>=1,
                            np.logical_and(deciphererror_df['pctError']>=0.01*(100-minfuzzyscore),
                                           deciphererror_df['pctBestReplace']>=0.5)),
decipherchars = deciphererror_df.index[errorprone]
improvementchars = deciphererror_df['ascii_replacement'].loc[errorprone].values
tmpasciimap = asciimap.copy()
for i,decipherchar in enumerate(decipherchars):
    asciimap['decipher'].loc[tmpasciimap['decipher'].values==decipherchar] = improvementchars[i]
# Repeating the previous code
# subsample of ciphertext
subset2_size = 2000
tuneindices = tune_df.index[subset_size:(subset2_size+subset_size)]

# parallel evaluation using Dask (big benefits for more CPUs)
par_compute2 = [delayed(finetunematch)(idx) for idx in tuneindices]
output_arrays = compute(*par_compute2, scheduler='processes')
sourcematch_array = [arrays[0] for arrays in output_arrays]
decipher_array = [arrays[1] for arrays in output_arrays]
fuzzyscores_array = np.array([arrays[2] for arrays in output_arrays])
sourcematch_array = list(compress(sourcematch_array, (fuzzyscores_array>=minfuzzyscore) ))
decipher_array = list(compress(decipher_array, (fuzzyscores_array>=minfuzzyscore) ))
cipher_array =  tune_df[subset_size:(subset_size+subset2_size)]['ciphertext']
cipher_array = list(compress(cipher_array, (fuzzyscores_array>=minfuzzyscore) ))

print('In 2nd subsample, ' + str(np.sum(fuzzyscores_array>=minfuzzyscore)) + ' of the ' + str(subset2_size) + ' have fuzzy.ratios > ' 
       + str(minfuzzyscore) + ' and will be used for fine-tuning')

# initialize asciiswitchmat, a 128x128 matrix with rows numbers corresponding to ascii characters
# that should be replaced by the ascii character of the column number
asciiswitchmat = np.zeros([128,128],dtype=int)
asciicount = np.zeros((128,), dtype=int)

for i, textstring in enumerate(decipher_array):
    asciiswitchmat += leven_replace_matrix(textstring,sourcematch_array[i])
    asciicount += np.histogram(string2ascii(textstring),np.arange(129))[0]

deciphererror_dict = {'ascii_replacement': [np.argmax(asciiswitchmat[i,...]) for i in range(128)],
                      'char_decipher': [chr(i) for i in range(128)],
                      'char_replacement': [chr(np.argmax(asciiswitchmat[i,...])) for i in range(128)],
                      'numError': np.sum(asciiswitchmat,axis=1),
                      'numReplace': [asciiswitchmat[i,np.argmax(asciiswitchmat[i,...])] for i in range(128)],
                      'totalObs': asciicount}

deciphererror_df = pd.DataFrame.from_dict(deciphererror_dict)
deciphererror_df['pctError'] = deciphererror_df['numError'] / deciphererror_df['totalObs']
deciphererror_df['pctBestReplace'] = deciphererror_df['numReplace'] / deciphererror_df['numError']

# correct the identified errors in character substitution
errorprone = np.logical_and(deciphererror_df['totalObs']>=1,
                            np.logical_and(deciphererror_df['pctError']>=0.1,
                                           deciphererror_df['pctBestReplace']>=0.8)),
decipherchars = deciphererror_df.index[errorprone]
improvementchars = deciphererror_df['ascii_replacement'].loc[errorprone].values
tmpasciimap = asciimap.copy()
for i,decipherchar in enumerate(decipherchars):
    asciimap['decipher'].loc[tmpasciimap['decipher'].values==decipherchar] = improvementchars[i]
print('After fine-tuning the sub-sample:')
gof = goodness_of_fit(cipher_array,asciimap=asciimap)
textstring = train_df[train_df['difficulty']==1]['ciphertext'].iloc[0]
print('FINAL DECIPHER:')
print(decipher_subst(textstring,asciimap))
%%time
test_df['plaintext'] = ['']*len(test_df)
# fill in plaintext for test data
c1indices = test_df[test_df['difficulty']==1].index
test_df.loc[c1indices,'plaintext'] = [decipher_subst(test_df['ciphertext'].loc[idx],asciimap) for idx in c1indices]
%%time
cpm_plaintext = char_per_million(chunks_plaintext)
asciitop = np.argsort(-1*cpm_plaintext)
asciitop_df = pd.DataFrame({'char':[chr(asciinum) for asciinum in asciitop],'char_per_million':cpm_plaintext[asciitop]},index=asciitop)

# we combine test and training data for greatest precision
cpm_train = char_per_million(train_df[train_df['difficulty']==2]['ciphertext'])
cpm_test =  char_per_million(test_df[test_df['difficulty']==2]['ciphertext'])
cpm_cipher = 0.67*cpm_test + 0.33*cpm_train

c2asciitop = np.argsort(-1*cpm_cipher)
c2asciitop_df = pd.DataFrame({'char':[chr(asciinum) for asciinum in c1asciitop],
                            'char_per_million':cpm_cipher[c2asciitop]},index=c2asciitop)

# create a bar chart to highlight the top 40 most common characters
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(np.arange(30),
       c2asciitop_df['char_per_million'].iloc[0:30],
       tick_label = ["'"+chr(asciinum)+"'" for asciinum in c2asciitop_df.index[0:30]] )
ax.plot(np.arange(30),asciitop_df['char_per_million'].iloc[0:30],color='red', marker='*', markersize=10)
ax.tick_params(axis='both',labelsize=16)
plt.ylabel('Occurrence per Million Characters',fontsize=16)
plt.title('Top 30 ASCII Characters: Ciphertext Difficulty=2',fontsize=18)
plt.show()
print("Of the 128 possible ASCII values, only "+ str(sum(cpm_cipher>1)) +" are used more than once per million")
%%time
asciimap2 = pd.DataFrame(asciitop, index=c2asciitop,columns=['decipher'])
# Go through the plain text from the category and find the matching entry
textstring = decipher_subst(train_df[train_df['difficulty']==2]['ciphertext'].iloc[0],asciimap2)
textcategory = train_df[train_df['difficulty']==2]['target'].iloc[0]
sourcematch = process.extractOne(textstring, sourcetext_dict, scorer = fuzz.ratio)

print('Found a match in source data with "fuzz.ratio" score of ' + str(fuzz.ratio(textstring,sourcematch[0])))
print('Source data found in category = ' + category_names[chunk_df['target'].iloc[sourcematch[2]]])
print('Original deciphering using character frequency:')
print('----------------')
print(textstring)
print(' ')
print('Matching text below:')
print('----------------')
print(sourcematch[0])
print(' ')
# subsample of ciphertext
subset_size = 2000
tune_df = train_df[np.logical_and(train_df['difficulty']==2,
                                    np.array([len(train_df['ciphertext'].iloc[idx]) for idx in range(len(train_df))])>100)]
tuneindices = tune_df.index[:subset_size]
        
# function to match to source data for same category and length greater than 100 characters
def finetunematch2(idx):
    textstring = decipher_subst(tune_df['ciphertext'].loc[idx],asciimap2)
    selectindices = np.logical_and(chunk_df['target']==tune_df['target'].loc[idx],chunk_df['length']==len(textstring))
    sourcematch = process.extractOne(textstring, 
                                     list(compress(chunks_plaintext, selectindices)), 
                                     scorer = fuzz.ratio)
    return sourcematch[0], textstring, sourcematch[1]

# parallel evaluation using Dask (big benefits for more CPUs)
par_compute = [delayed(finetunematch2)(idx) for idx in tuneindices]
output_arrays = compute(*par_compute, scheduler='processes')
# select those with relatively high fuzzy.ratio scores
minfuzzyscore = 80
sourcematch_array = [arrays[0] for arrays in output_arrays]
decipher_array = [arrays[1] for arrays in output_arrays]
fuzzyscores_array = np.array([arrays[2] for arrays in output_arrays])
sourcematch_array = list(compress(sourcematch_array, (fuzzyscores_array>=minfuzzyscore) ))
decipher_array = list(compress(decipher_array, (fuzzyscores_array>=minfuzzyscore) ))
cipher_array =  tune_df[:subset_size]['ciphertext']
cipher_array = list(compress(cipher_array, (fuzzyscores_array>=minfuzzyscore) ))
print('In subsample, ' + str(np.sum(fuzzyscores_array>=minfuzzyscore)) + ' of the ' + str(subset_size) + ' have fuzzy.ratios > ' 
       + str(minfuzzyscore) + ' and will be used for fine-tuning')
print(' ')
print('Before fine-tuning the sub-sample:')
gof = goodness_of_fit(cipher_array,asciimap=asciimap2)
print('Average fuzzy.ratio = ' + str(np.mean(fuzzyscores_array[fuzzyscores_array>=minfuzzyscore])))
# initialize asciiswitchmat, a 128x128 matrix with rows numbers corresponding to ascii characters
# that should be replaced by the ascii character of the column number
asciiswitchmat = np.zeros([128,128],dtype=int)
asciicount = np.zeros((128,), dtype=int)

for i, textstring in enumerate(decipher_array):
    asciiswitchmat += leven_replace_matrix(textstring,sourcematch_array[i])
    asciicount += np.histogram(string2ascii(textstring),np.arange(129))[0]

deciphererror_dict2 = {'ascii_replacement': [np.argmax(asciiswitchmat[i,...]) for i in range(128)],
                      'char_decipher': [chr(i) for i in range(128)],
                      'char_replacement': [chr(np.argmax(asciiswitchmat[i,...])) for i in range(128)],
                      'numError': np.sum(asciiswitchmat,axis=1),
                      'numReplace': [asciiswitchmat[i,np.argmax(asciiswitchmat[i,...])] for i in range(128)],
                      'totalObs': asciicount}

deciphererror_df2 = pd.DataFrame.from_dict(deciphererror_dict2)
deciphererror_df2['pctError'] = deciphererror_df2['numError'] / deciphererror_df2['totalObs']
deciphererror_df2['pctBestReplace'] = deciphererror_df2['numReplace'] / deciphererror_df2['numError']  
deciphererror_df2.sort_values(by=['numError'],ascending=False,inplace=True)
# correct the identified errors in character substitution
errorprone = np.logical_and(deciphererror_df2['totalObs']>=1,
                            np.logical_and(deciphererror_df2['pctError']>=0.1,
                                           deciphererror_df2['pctBestReplace']>=0.5)),
decipherchars = deciphererror_df2.index[errorprone]
improvementchars = deciphererror_df2['ascii_replacement'].loc[errorprone].values
tmpasciimap = asciimap2.copy()
for i,decipherchar in enumerate(decipherchars):
    asciimap2['decipher'].loc[tmpasciimap['decipher'].values==decipherchar] = improvementchars[i]
subset2_size = 2000
tuneindices = tune_df.index[subset_size:(subset2_size+subset_size)]

# parallel evaluation using Dask (big benefits for more CPUs)
par_compute2 = [delayed(finetunematch2)(idx) for idx in tuneindices]
output_arrays = compute(*par_compute2, scheduler='processes')


# select those with relatively high fuzzy.ratio scores
minfuzzyscore = 80
sourcematch_array = [arrays[0] for arrays in output_arrays]
decipher_array = [arrays[1] for arrays in output_arrays]
fuzzyscores_array = np.array([arrays[2] for arrays in output_arrays])
sourcematch_array = list(compress(sourcematch_array, (fuzzyscores_array>=minfuzzyscore) ))
decipher_array = list(compress(decipher_array, (fuzzyscores_array>=minfuzzyscore) ))
cipher_array =  tune_df[subset_size:(subset_size+subset2_size)]['ciphertext']
cipher_array = list(compress(cipher_array, (fuzzyscores_array>=minfuzzyscore) ))


print('In 2nd subsample, ' + str(np.sum(fuzzyscores_array>=minfuzzyscore)) + ' of the ' + str(subset2_size) + ' have fuzzy.ratios > ' 
       + str(minfuzzyscore) + ' and will be used for fine-tuning')

# initialize asciiswitchmat, a 128x128 matrix with rows numbers corresponding to ascii characters
# that should be replaced by the ascii character of the column number
asciiswitchmat = np.zeros([128,128],dtype=int)
asciicount = np.zeros((128,), dtype=int)

for i, textstring in enumerate(decipher_array):
    asciiswitchmat += leven_replace_matrix(textstring,sourcematch_array[i])
    asciicount += np.histogram(string2ascii(textstring),np.arange(129))[0]

deciphererror_dict = {'ascii_replacement': [np.argmax(asciiswitchmat[i,...]) for i in range(128)],
                      'char_decipher': [chr(i) for i in range(128)],
                      'char_replacement': [chr(np.argmax(asciiswitchmat[i,...])) for i in range(128)],
                      'numError': np.sum(asciiswitchmat,axis=1),
                      'numReplace': [asciiswitchmat[i,np.argmax(asciiswitchmat[i,...])] for i in range(128)],
                      'totalObs': asciicount}

deciphererror_df = pd.DataFrame.from_dict(deciphererror_dict)
deciphererror_df['pctError'] = deciphererror_df['numError'] / deciphererror_df['totalObs']
deciphererror_df['pctBestReplace'] = deciphererror_df['numReplace'] / deciphererror_df['numError']

# correct the identified errors in character substitution
errorprone = np.logical_and(deciphererror_df['totalObs']>=2,
                            np.logical_and(deciphererror_df['pctError']>=0.2,
                                           deciphererror_df['pctBestReplace']>=0.66)),
decipherchars = deciphererror_df.index[errorprone]
improvementchars = deciphererror_df['ascii_replacement'].loc[errorprone].values
tmpasciimap = asciimap2.copy()
for i,decipherchar in enumerate(decipherchars):
    asciimap2['decipher'].loc[tmpasciimap['decipher'].values==decipherchar] = improvementchars[i]
print('After fine-tuning the sub-sample:')
gof = goodness_of_fit(cipher_array,asciimap=asciimap2)
textstring = train_df[train_df['difficulty']==2]['ciphertext'].iloc[0]
print('FINAL DECIPHER:')
print(decipher_subst(textstring,asciimap2))
%%time
c2indices = test_df[test_df['difficulty']==2].index
test_df.loc[c2indices,'plaintext'] = [decipher_subst(test_df['ciphertext'].loc[idx],asciimap2) for idx in c2indices]
# we combine test and training data for greatest precision
cpm_train = char_per_million(train_df[train_df['difficulty']==3]['ciphertext'])
cpm_test =  char_per_million(test_df[test_df['difficulty']==3]['ciphertext'])
cpm_cipher = 0.67*cpm_test + 0.33*cpm_train

c3asciitop = np.argsort(-1*cpm_cipher)
c3asciitop_df = pd.DataFrame({'char':[chr(asciinum) for asciinum in c3asciitop],
                            'char_per_million':cpm_cipher[c3asciitop]},index=c3asciitop)

# create a bar chart to highlight the top 40 most common characters
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(np.arange(30),
       c3asciitop_df['char_per_million'].iloc[0:30],
       tick_label = ["'"+chr(asciinum)+"'" for asciinum in c3asciitop_df.index[0:30]] )
ax.plot(np.arange(30),asciitop_df['char_per_million'].iloc[0:30],color='red', marker='*', markersize=10)
ax.tick_params(axis='both',labelsize=16)
plt.ylabel('Occurrence per Million Characters',fontsize=16)
plt.title('Top 30 ASCII Characters: Ciphertext Difficulty=3',fontsize=18)
plt.legend(['Distribution for Top Characters in Source Data','Distribution for Difficulty=3'],fontsize=16)
plt.show()
# Go through the plain text from the category and find the matching entry
textstring = decipher_subst(train_df[train_df['difficulty']==3]['ciphertext'].iloc[0],asciimap2)
textcategory = train_df[train_df['difficulty']==3]['target'].iloc[0]
sourcematch = process.extractOne(textstring, sourcetext_dict, scorer = fuzz.ratio)

print('Found a match in source data with "fuzz.ratio" score of ' + str(fuzz.ratio(textstring,sourcematch[0])))
print('Source data found in category = ' + category_names[chunk_df['target'].iloc[sourcematch[2]]])
print('Original deciphering of difficulty #3 ciphertext of using mapping from cipher #2:')
print('----------------')
print(textstring)
print(' ')
print('Matching text below:')
print('----------------')
print(sourcematch[0])
print(' ')
# subsample of ciphertext
subset_size = 400

reversemap2 = asciimap2.copy()
reversemap2.drop_duplicates('decipher',inplace=True)
reversemap2.reset_index(inplace=True)
reversemap2.rename(index=str,columns={"decipher":"index","index":"decipher"},inplace=True)
reversemap2.set_index('index',inplace=True)
reversemap2.head()

transform_ct = np.zeros((128,), dtype=int)
transform_out = np.zeros((128,), dtype=int)
stable_ct = np.zeros((128,), dtype=int)
for idx in tqdm(range(subset_size)):
    textstring = decipher_subst(train_df[train_df['difficulty']==3]['ciphertext'].iloc[idx],asciimap2)
    textcategory = train_df[train_df['difficulty']==3]['target'].iloc[idx]
    sourcematch = process.extractOne(textstring, 
                                     list(compress(chunks_plaintext,chunks_target==textcategory)), 
                                     scorer = fuzz.ratio)
    s1 = string2ascii(decipher_subst(sourcematch[0],reversemap2))
    s2 = string2ascii(train_df[train_df['difficulty']==3]['ciphertext'].iloc[idx])
    if len(s1)==len(s2):
        transform_ct += np.histogram(np.array(s1[(s1-s2)>0],dtype=int),np.arange(129))[0]
        transform_out += np.histogram(np.array(s2[(s1-s2)>0],dtype=int),np.arange(129))[0]
        stable_ct += np.histogram(np.array(s1[(s1-s2)==0],dtype=int),np.arange(129))[0]

transform_pct = np.divide(transform_ct, (transform_ct + stable_ct+1) )

enoughobs = np.arange(128)[np.logical_and((transform_ct+stable_ct)>5,np.arange(128)>40)]
fig, ax = plt.subplots(figsize=(20, 4))
ax.bar(np.arange(len(enoughobs)),
        transform_pct[enoughobs],
              tick_label = ["'"+chr(asciinum)+"'" for asciinum in enoughobs] )
plt.title('Characters Transformed by Cipher Number 3',fontsize=16)
plt.show()
c3mod = set()
modascii = np.union1d(np.arange(65,91),np.arange(97,123))
for i in modascii:
    c3mod.add(i)
%%time
train_lengths = np.array([len(cipherstring) for cipherstring in train_df['ciphertext']])
c3indices = train_df.index[np.logical_and(train_df['difficulty']==3,train_lengths>290)]

numsamples = 50
c3modmat = np.nan*np.zeros((numsamples,300))
s2startmat = np.nan*np.zeros((numsamples,300)) 
for i,idx in enumerate(c3indices[0:numsamples]):
    textstring = decipher_subst(train_df['ciphertext'].loc[idx],asciimap2)
    textcategory = train_df['target'].loc[idx]
    selectindices = np.logical_and(chunk_df['target']==textcategory,chunk_df['length']>290)
    sourcematch = process.extractOne(textstring,
                                     list(compress(chunks_plaintext, selectindices)),
                                     scorer = fuzz.ratio)
    # Look at the differences for the modified characters
    s1 = string2ascii(decipher_subst(sourcematch[0],reversemap2))
    s2 = string2ascii(train_df['ciphertext'].loc[idx])
    s2mods = [(snum in c3mod) for snum in s2]
    ds = (s1[s2mods]-s2[s2mods])
    # append to matrix
    c3modmat[i,0:len(ds)] = ds
    s2startmat[i,0:len(ds)] = s2[s2mods]
    
    
# with only 26 letters, we cycle back to the beginning
c3modmat[c3modmat<0] = c3modmat[c3modmat<0]+26

fig, ax = plt.subplots(figsize=(15, 5))
#ax.plot(np.arange(300),c3modmat.transpose())
ax.matshow(c3modmat[0:40,0:80])
plt.xlabel('Character Changes',fontsize=16)
plt.ylabel('Ciphertext Samples',fontsize=16)
plt.title("EUREKA! A CONSISTENT PATTERN!",fontsize=22)
plt.show()
modeadjust,_ = stats.mode(c3modmat,axis=0,nan_policy='omit')
modeadjust = np.squeeze(modeadjust.data)
modeadjust = np.trim_zeros(modeadjust)
startadj = 91
print('Modal adjustment to alphabetic characters')
print(ascii2string(startadj-np.array(modeadjust,dtype=int)))
print('Chars 0-19')
print(ascii2string(startadj-np.array(modeadjust[0:19],dtype=int)))
print('Chars 19-38')
print(ascii2string(startadj-np.array(modeadjust[19:37],dtype=int)))
print('Key = ')
modeadjust0 = [ 19.,  22.,  15.,  22.,  13.,   1.,  21.,  12.,   6.,  24.,  19.,
        22.,   1.,  20.,   26.,  18.,  13.,  22.,   8.]
print(ascii2string(startadj-np.array(modeadjust0,dtype=int)))
def decipher_rolling(textstring,c3mod,c3adjust):
    inarray = string2ascii(textstring)
    outarray = np.zeros(len(inarray),dtype=np.int8)
    outarray[:] = inarray
    # check to see if any substitutions are needed
    modchars = [(snum in c3mod) for snum in inarray]
    nmods = sum(modchars)
    if nmods>0:
        modchars = np.squeeze(np.argwhere(modchars))
        lowermods = outarray[modchars] > 95
        charadj = outarray[modchars] - 65 - 32*lowermods
        charadj = (charadj + c3adjust[:nmods])%26
        outarray[modchars] = 65+32*lowermods+charadj
    return ascii2string(outarray)

c3adjust = np.tile(modeadjust0,16)
c3adjust = c3adjust[0:300]
textstring = train_df[train_df['difficulty']==3]['ciphertext'].iloc[0]
print('FINAL DECIPHER:')
print(decipher_subst(decipher_rolling(textstring,c3mod,c3adjust),asciimap2))
%%time
c3indices = test_df[test_df['difficulty']>2].index
test_df.loc[c3indices,'plaintext'] = [decipher_subst(decipher_rolling(test_df['ciphertext'].loc[idx],c3mod,c3adjust),asciimap2) for idx in c3indices]
def minleven(textstring,selectdataframe):
    arrayofstrings = selectdataframe['plaintext'].values
    ldistances = [leven.distance(textstring,compstring) for compstring in arrayofstrings]
    bestfitref = np.argmin(ldistances)
    bestfittext = selectdataframe['plaintext'].iloc[bestfitref]
    bestfitindex = selectdataframe.index[bestfitref]
    return bestfittext, bestfitindex

def char_count(textstring):
    return np.histogram(string2ascii(textstring),np.arange(129))[0]
    
# calculate ascii character frequency per million characters
def char_per_300(stringsarray):
    asciicount = np.zeros((128,), dtype=int)
    for k,textdata in enumerate(stringsarray):
        asciicount += char_count(textdata)
    totalchars = np.sum(asciicount)
    return np.multiply(asciicount,np.divide(300.0,totalchars))

def similarity_score(x, y):
    #ss = 1.0 - np.divide(np.linalg.norm(x - y),np.linalg.norm(y))
    ss = np.sum(np.multiply(x,y))
    return ss
%%time
# major speedup with a pseudo-hash
cp300 = char_per_300(chunks_plaintext)
test_df['Predicted'] = 1
test_df['Levenshtein'] = np.nan
test_df['lookupscore'] = [similarity_score(char_count(textstring),cp300) for textstring in test_df['plaintext'].values ]
chunk_df['lookupscore'] = [similarity_score(char_count(textstring),cp300) for textstring in chunk_df['plaintext'].values ]
def exacttestmatch(idx):
    textstring = test_df['plaintext'].loc[idx]
    selectindices = np.argwhere(chunk_df['lookupscore']==test_df['lookupscore'].loc[idx])
    if len(selectindices)==1:
        matchtext = chunk_df['plaintext'].loc[selectindices[0]].values[0]
        matchtarget = chunk_df['target'].loc[selectindices[0]].values[0]
        matchdist = leven.distance(textstring,matchtext)
    else:
        matchtarget = np.nan
        matchdist = 300
    return matchtarget, matchdist


for idx in tqdm(test_df.index):
    matchtarget, matchdist = exacttestmatch(idx)
    if matchdist == 0:
        test_df.loc[idx,'Predicted'] = matchtarget
        test_df.loc[idx,'Levenshtein'] = matchdist
        
print(str(np.sum(test_df['Levenshtein']==0))+"/"+str(len(test_df))+" exact matches")
submission_df = test_df.copy()
submission_df.set_index('Id',inplace=True)
submission_df.drop(['difficulty','plaintext','ciphertext','length','Levenshtein','lookupscore'],
                   axis=1,inplace=True)
submission_df['Predicted'] = pd.to_numeric(submission_df['Predicted'],downcast='integer')
submission_df.to_csv('submission.csv')