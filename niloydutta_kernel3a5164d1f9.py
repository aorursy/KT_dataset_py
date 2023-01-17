%%javascript
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
});
import requests
import time

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import re
import glob
import random
import seaborn as sns
import string

from IPython.display import clear_output

# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# http://www.nltk.org/howto/wordnet.html

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.wsd import lesk
# Location of test/train data files on local computer, data downloaded directly from Stanford source[2]
#test_dir = '/Users/philiposborne/Documents/Written Notes/Learning Notes/IMDB Reviews/IMDB Data/test'
#train_dir = '/Users/philiposborne/Documents/Written Notes/Learning Notes/IMDB Reviews/IMDB Data/train'

data = pd.read_csv('../input/imdb_master.csv',encoding="latin-1")

# Select only training data
data = data[data['type']=='train'].reset_index(drop=True)
print('Number of comments in data:', len(data))

data = data[0:1000]

print('Number of comments left in data after removal:', len(data))
train_data = data
# Data import written as a function:
# Replace test and train dir with correct path for file saved on local computer
# Data files are downloaded from reference link above where main file name is changed to IMDB Data

# This function converts the raw files form the original Stanford source into csv files.
"""
def IMDB_to_csv(directory):    
    data = pd.DataFrame()
    
    for filename in glob.glob(str(directory)+'/neg/*.txt'):
        with open(filename, 'r',  encoding="utf8") as f:
            content = f.readlines()
            content_table = pd.DataFrame({'id':filename.split('_')[0].split('/')[-1],'rating':filename.split('_')[1].split('.')[0],'pol':'neg', 'text':content})
        data = data.append(content_table)
        
    for filename in glob.glob(str(directory)+'/pos/*.txt'):
        with open(filename, 'r',  encoding="utf8") as f:
            content = f.readlines()
            content_table = pd.DataFrame({'id':filename.split('_')[0].split('/')[-1],'rating':filename.split('_')[1].split('.')[0],'pol':'pos', 'text':content})
        data = data.append(content_table)
    data = data.sort_values(['pol','id'])
    data = data.reset_index(drop=True)
    #data['rating_norm'] = (data['rating'] - data['rating'].min())/( data['rating'].max() - data['rating'].min() )

    return(data)

train_data = IMDB_to_csv(train_dir)
"""
train_data.columns = ['id', 'dataset', 'text', 'pol','file']
train_data.head()
train_data['text'][0].split('.')
train_data['text'][0].split('.')[0]
len(train_data['text'][0].split('.'))
train_data['text'][0].split('.')[8]
train_data_sent = pd.DataFrame()

start_time = time.time()
for index in train_data.index:
    data_row = train_data.iloc[index,:]

    for sent_id in range(0,len(data_row['text'].split('.'))-1):
        sentence = data_row['text'].split('.')[sent_id]
        # Form a row in a dataframe for this setence that captures the words and keeps ids and polarity scores
        # We must pass an arbitrary index which we then reset to show unique numbers
        sentence_row = pd.DataFrame({
                                     'id':data_row['id'],
                                     'pol':data_row['pol'],
                                     'sent_id':sent_id,
                                     'sentence':sentence}, index = [index]) 
        
        # Form full table that has rows for all sentences
        train_data_sent = train_data_sent.append(sentence_row)
    
    
    # Outputs progress of main loop, see:
    clear_output(wait=True)
    print('Proportion of comments completed:', np.round(index/len(train_data),4)*100,'%')
    
end_time = time.time()
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')
# Reset index so that each index value is a unique number
train_data_sent = train_data_sent.reset_index(drop=True)
        
train_data_sent.head()
train_data_sent['sentence_clean'] = train_data_sent['sentence'].str.replace('[{}]'.format(string.punctuation), '')
train_data_sent['sentence_clean'] = train_data_sent['sentence_clean'].str.lower()

train_data_sent['sentence_clean'] = '<s ' + train_data_sent['sentence_clean']
train_data_sent['sentence_clean'] = train_data_sent['sentence_clean'] + ' /s>'

train_data_sent.head()
text = train_data_sent['sentence_clean']
text_list = " ".join(map(str, text))
text_list[0:100]
word_list = pd.DataFrame({'words':text.str.split(' ', expand = True).stack().unique()})
word_count_table = pd.DataFrame()
for n,word in enumerate(word_list['words']):
    # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
    # e.g. 'ear' would be counted in each appearance of the word 'year'
    word_count = len(re.findall(' ' + word + ' ', text_list))  
    word_count_table = word_count_table.append(pd.DataFrame({'count':word_count}, index=[n]))
    
    clear_output(wait=True)
    print('Proportion of words completed:', np.round(n/len(word_list),4)*100,'%')

word_list['count'] = word_count_table['count']
# Remove the count for the start and end of sentence notation so 
# that these do not inflate the other probabilities
word_list['count'] = np.where(word_list['words'] == '<s' , 0,
                     np.where(word_list['words'] == '/s>', 0,
                     word_list['count']))
word_list['prob'] = word_list['count']/sum(word_list['count'])
word_list.head()
unigram_table = pd.DataFrame()

start_time = time.time()
# Loop through each sentence
# REMOVE ROW LIMIT FOR FULL RUN
for index in train_data_sent[0:200].index:
    data_row = train_data_sent.iloc[index,:]

    sent_probs = pd.DataFrame()
    # Go through each word in the sentence, lookup the probability of the word and 
    # then find the mulitplicitive product of all probabilities in the sentence.
    for n,word in enumerate(data_row['sentence_clean']):
        sent_probs = sent_probs.append(pd.DataFrame({'prob':word_list[ word_list['words']==word]['prob']}, index = [n]))
    unigram = sent_probs['prob'].prod(axis=0)
    
    # Create a list of unigram calculation for each sentence
    unigram_table = unigram_table.append(pd.DataFrame({'unigram':unigram},index = [index]))
    
    clear_output(wait=True)
    print('Proportion of sentences completed:', np.round(index/len(train_data_sent),4)*100,'%')
        
end_time = time.time()
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')

train_data_sent['unigram'] = unigram_table['unigram']
base_time = end_time-start_time
unigram_table.head(10)
unigram_table_log = pd.DataFrame()

start_time_log = time.time()
# Loop through each sentence
# REMOVE ROW LIMIT FOR FULL RUN
for index in train_data_sent[0:200].index:
    data_row = train_data_sent.iloc[index,:]

    sent_probs = pd.DataFrame()
    # Go through each word in the sentence, lookup the probability of the word and 
    # then find the mulitplicitive product of all probabilities in the sentence.
    for n,word in enumerate(data_row['sentence_clean']):
        log_prob = np.log10(word_list[ word_list['words']==word]['prob'])
        sent_probs = sent_probs.append(pd.DataFrame({'log_prob':log_prob}, index = [n]))
        
    unigram_log = sum(sent_probs['log_prob'])
    
    # Create a list of unigram calculation for each sentence
    unigram_table_log = unigram_table.append(pd.DataFrame({'unigram_log':unigram_log},index = [index]))
                                         
    clear_output(wait=True)
    print('Proportion of sentences completed:', np.round(index/len(train_data_sent),4)*100,'%')
                                                                   
end_time_log = time.time()
print('Total run time = ', np.round(end_time_log-start_time_log,2)/60, ' minutes')

#train_data_sent['unigram_log'] = unigram_table_log['unigram_log']
log_time = end_time_log - start_time_log
print('The log base 10 method takes approximately ', np.round((log_time)/base_time,4)*100, '% of the time of the orginal calculation.')
word_1 = 'to'
word_2 = 'a'

prob_word_1 = word_list[word_list['words'] == word_1]['prob'].iloc[0]
prob_word_2 = word_list[word_list['words'] == word_2]['prob'].iloc[0]

unigram_prob = prob_word_1*prob_word_2

print('The unigram probability of the word "a" occuring given the word "to" was the previous word is: ', np.round(unigram_prob,10))
word_1 = ' ' + str('to') + ' '
word_2 = str('a') + ' '

bigram_prob = len(re.findall(word_1 + word_2, text_list)) / len(re.findall(word_1, text_list)) 

print('The probability of the word "a" occuring given the word "to" was the previous word is: ', np.round(bigram_prob,5))
word_1 = ' ' + str('has') + ' '
word_2 = str('a') + ' '

bigram_prob = len(re.findall(word_1 + word_2, text_list)) / len(re.findall(word_1, text_list)) 

print('The probability of the word "a" occuring given the word "has" was the previous word is: ', np.round(bigram_prob,5))
W_W_Matrix = pd.DataFrame({'words': word_list['words']})

start_time = time.time()


# Add limits to number of columns/rows so this doesn't run for ages
column_lim = 1000
#column_lim = len(W_W_Matrix)
row_lim = 10
#row_lim = len(W_W_Matrix)

for r, column in enumerate(W_W_Matrix['words'][0:column_lim]):
    
    prob_table = pd.DataFrame()
    for i, row in enumerate(W_W_Matrix['words'][0:row_lim]):

        word_1 = ' ' + str(row) + ' '
        word_2 = str(column) + ' '

        if len(re.findall(word_1, text_list)) == 0:
            prob = pd.DataFrame({'prob':[0]}, index=[i])
        else:
            prob = pd.DataFrame({'prob':[len(re.findall(word_1 + word_2, text_list)) / len(re.findall(word_1, text_list)) ]}, index=[i])
        
        prob_table = prob_table.append(prob)
    W_W_Matrix[str(column)] = prob_table['prob']
    
    # Outputs progress of main loop, see:
    clear_output(wait=True)
    print('Proportion of column words completed:', np.round(r/len(W_W_Matrix[0:column_lim]),2)*100,'%')
    
end_time = time.time()
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')

W_W_Matrix[W_W_Matrix['a'] >= 0]
for i in range(0,row_lim):
    plt.bar(W_W_Matrix.iloc[i,1:].sort_values(ascending=False)[1:10].index,W_W_Matrix.iloc[i,1:].sort_values(ascending=False)[1:10].values)
    plt.title('Most Common Words that Follow the word: ' +str(W_W_Matrix.iloc[i,0]))
    plt.show()
W_W_Matrix = pd.DataFrame({'words': word_list['words']})

start_time = time.time()

text_list = " ".join(map(str, text))

# Increasing these take significant time to run but provide more realistic sentences
num_sentences = 1
sentence_word_limit = 1

#extract start and end of sentence notation so that they are always included
sentence_forms = W_W_Matrix[(W_W_Matrix['words']=='<s') | (W_W_Matrix['words']=='/s>')]['words']

sentences_output = pd.DataFrame()
for sample in range(0,num_sentences):
    
    sentence = pd.DataFrame()
    
    for i in range(0,sentence_word_limit):
        # if this is the first word, fix it to be start of sentence notation else take output of previous iteration
        if (i==0):
            current_word = str('<s')
        # Randomly select first word after sentence start
        elif (i==1):
            current_word = str(W_W_Matrix[(W_W_Matrix['words']!='<s') ]['words'].sample(1, axis=0).iloc[0])
        else:
            current_word = next_word
        
        sentence['word_'+str(i)] = [current_word]
        # if we have reached end of sentence, add this sentence to output table and break loop to start new sentence
        if (current_word==str('/s>')):
            sentences_output = sentences_output.append(sentence)
            break   
        else:
            
            prob_table = pd.DataFrame()

            # randomly select other words form rest of list
            for r, column in enumerate(W_W_Matrix[(W_W_Matrix['words']!='<s') ]['words'].reset_index(drop=True)):

                next_words = str(column)



                if len(re.findall(' ' + current_word + ' ', text_list)) == 0:
                    prob = pd.DataFrame({'word':str(column),
                        'prob':[0]}, index=[i])
                else:
                    prob = pd.DataFrame({'word':str(column),
                        'prob':[len(re.findall(' ' + current_word + ' ' + next_words + ' ', text_list)) / len(re.findall(' ' + current_word + ' ', text_list)) ]}, index=[i])

                prob_table = prob_table.append(prob)
                # We can reduce the probability of the sentence ending so that we return longer sentences
                reduce_end_prob = 0.5
                prob_table['prob'] = np.where(prob_table['word']=='/s>', prob_table['prob']*reduce_end_prob,prob_table['prob'])
                # next word is most probable of this given the current word
                next_word = prob_table[prob_table['prob'] == max(prob_table['prob'])]['word'].reset_index(drop=True).iloc[0]
                
                # Outputs progress of main loop:
                clear_output(wait=True)
                print("Sentence number: ",sample+1)
                print("Words completed in current sentence:",i+1)
                print('Proportion of column words completed:', np.round(r/len(W_W_Matrix),2)*100,'%')

        
end_time = time.time()
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')

for i in range(1,num_sentences):
    print('Sentence ',i,':',sentences_output.iloc[i].values)
word_1 = str('movie')
word_2 = str('/s>') 

bigram_prob = len(re.findall(' ' + word_1 + ' ' + word_2 + ' ', text_list)) / len(re.findall(' ' + word_1 + ' ', text_list))


print('The probability of the word "',word_2,'" occuring given the word "',word_1,'" was the previous word is: ', (bigram_prob))
word_1 = str('to')
word_2 = str('a') 
word_3 = str('movie')

trigram_prob = (len(re.findall(' ' + word_1 + ' ' + word_2 + ' ' + word_3 + ' ', text_list)) / 
                    len(re.findall(' ' + word_1 + ' ' + word_2, text_list)))


print('The probability of the word "',word_3,'" occuring given the word "',word_1,'" and "',word_2,'" were the previous two words is: ', (trigram_prob))
word_1 = str('to')
word_2 = str('a') 
word_3 = str('film')

trigram_prob = (len(re.findall(' ' + word_1 + ' ' + word_2 + ' ' + word_3 + ' ', text_list)) / 
                    len(re.findall(' ' + word_1 + ' ' + word_2, text_list)))


print('The probability of the word "',word_3,'" occuring given the word "',word_1,'" and "',word_2,'" were the previous two words is: ', (trigram_prob))
p = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
H = [-(p*np.log2(p) + (1-p)*np.log(1-p)) for p in p]
# Replace nan output with 0 
H = [0 if math.isnan(x) else x for x in H]

plt.plot(p,H)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1])
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Entropy (H(p))')
plt.title('The Entropy of a Bias Coin as \n the Probabilitiy of Heads Varies')
sent_1 = text.iloc[0]
sent_2 = text.iloc[1]

print('Sentence 1', sent_1)
print('--.--.--.--.--.--.--.--')
print('Sentence 2', sent_2)
data_prob = word_list[['words','count','prob']]
data_prob.head()
def entropy(sentence, data_prob):
    entropy_table = pd.DataFrame()
    for n,word in enumerate(sentence.split(' ')):
        # log2(0) provide nan so return 0 instead
        if ((data_prob[data_prob['words']==word]['prob'].iloc[0]) == 0):
            entropy = 0
        else:
            prob = data_prob[data_prob['words']==word]['prob'].iloc[0]
            entropy = prob*np.log2(prob)
        entropy_table = entropy_table.append(pd.DataFrame({'word':word,
                                                            'entropy':entropy}, index = [n]))
    phrase_entropy = -1*sum(entropy_table['entropy'])
    return(phrase_entropy)
sent_1_entropy = entropy(sent_1,data_prob)
sent_2_entropy = entropy(sent_2,data_prob)

print('Sentence 1: ', sent_1)
print('Sentence 1 entropy: ', np.round(sent_1_entropy,5))
print('Per-word Sentence 1 entropy: ', np.round(sent_1_entropy/len(sent_1.split(' ')),5))

print('--.--.--.--.--.--.--.--')
print('Sentence 2: ', sent_2)
print('Sentence 2 entropy: ', np.round(sent_2_entropy,5))
print('Per-word Sentence 2 entropy: ', np.round(sent_2_entropy/len(sent_2.split(' ')),5))

sent_1_perplex = 2**sent_1_entropy
sent_2_perplex = 2**sent_2_entropy

print('Sentence 1: ', sent_1)
print('Sentence 1 entropy: ', np.round(sent_1_entropy,5))
print('Per-word Sentence 1 entropy: ', np.round(sent_1_entropy/len(sent_1.split(' ')),5))
print('Sentence 1 Perplexity: ', sent_1_perplex)

print('--.--.--.--.--.--.--.--')
print('Sentence 2: ', sent_2)
print('Sentence 2 entropy: ', np.round(sent_2_entropy,5))
print('Per-word Sentence 2 entropy: ', np.round(sent_2_entropy/len(sent_2.split(' ')),5))
print('Sentence 2 Perplexity: ', sent_2_perplex)

sent_1_prob = (1/sent_1_perplex)**len(sent_1.split(' '))
sent_2_prob = (1/sent_2_perplex)**len(sent_2.split(' '))

print('Sentence 1 Probability: ', '%0.10f' % sent_1_prob)
print('Sentence 2 Probability: ', '%0.10f' % sent_2_prob  )
train_data_sent.head()
corpus = train_data_sent['sentence_clean'][:int(np.round(len(train_data_sent)*0.9,0))]
test = train_data_sent['sentence_clean'][int(np.round(len(train_data_sent)*0.9,0))+1:]

corpus_list = " ".join(map(str, corpus))
test_list = " ".join(map(str, test))

# Corpus word probabilities
corpus_word_list = pd.DataFrame({'words':corpus.str.split(' ', expand = True).stack().unique()})
corpus_word_count_table = pd.DataFrame()
for n,word in enumerate(corpus_word_list['words']):
    # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
    # e.g. 'ear' would be counted in each appearance of the word 'year'
    corpus_word_count = len(re.findall(' ' + word + ' ', corpus_list))  
    corpus_word_count_table = corpus_word_count_table.append(pd.DataFrame({'count':corpus_word_count}, index=[n]))
    
    clear_output(wait=True)
    print('Proportion of words completed:', np.round(n/len(corpus_word_list),4)*100,'%')

corpus_word_list['count'] = corpus_word_count_table['count']
# Remove the count for the start and end of sentence notation so 
# that these do not inflate the other probabilities
corpus_word_list['count'] = np.where(corpus_word_list['words'] == '<s' , 0,
                     np.where(corpus_word_list['words'] == '/s>', 0,
                     corpus_word_list['count']))
corpus_word_list['prob'] = corpus_word_list['count']/sum(corpus_word_list['count'])




corpus_word_list.head()
# Test set word probabilities
test_word_list = pd.DataFrame({'words':test.str.split(' ', expand = True).stack().unique()})
test_word_count_table = pd.DataFrame()
for n,word in enumerate(test_word_list['words']):
    # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
    # e.g. 'ear' would be counted in each appearance of the word 'year'
    test_word_count = len(re.findall(' ' + word + ' ', test_list))  
    test_word_count_table = test_word_count_table.append(pd.DataFrame({'count':test_word_count}, index=[n]))
    
    clear_output(wait=True)
    print('Proportion of words completed:', np.round(n/len(test_word_list),4)*100,'%')

test_word_list['count'] = test_word_count_table['count']
# Remove the count for the start and end of sentence notation so 
# that these do not inflate the other probabilities
test_word_list['count'] = np.where(test_word_list['words'] == '<s' , 0,
                     np.where(test_word_list['words'] == '/s>', 0,
                     test_word_list['count']))
test_word_list['prob'] = test_word_list['count']/sum(test_word_list['count'])


test_word_list.head()
# Merge corpus counts to test set and replace missing values with 0
test_word_list_2 = test_word_list.merge(corpus_word_list[['words','count']], how='left', on = 'words')
test_word_list_2['count_y'].fillna(0, inplace=True)

test_word_list_2.head()
print('Percentage of words in test set that are not contained in corpus',len(test_word_list_2[test_word_list_2['count_y']==0])/len(test_word_list_2)*100,'%')
# Extract missing words from training set
missing_words = test_word_list_2[test_word_list_2['count_y']==0]
missing_words = missing_words[(missing_words['words']!='<s')&(missing_words['words']!='/s>')]
missing_words = missing_words[['words']]
missing_words['count'] = 0
missing_words['prob'] = 0
missing_words.head()
# Add missing words onto end of corpus word list and apply laplace +1 smoothing
corpus_word_list_fixed = corpus_word_list.append(missing_words)

corpus_word_list_fixed['count+1'] = corpus_word_list_fixed['count']+1
corpus_word_list_fixed['prob+1'] = corpus_word_list_fixed['count+1']/sum(corpus_word_list_fixed['count+1'])
corpus_word_list_fixed.head()
# Plot distribution before and after Laplace +1 Smoothing
sns.distplot(corpus_word_list_fixed[corpus_word_list_fixed['count']<=15]['count'], label='Before')
sns.distplot(corpus_word_list_fixed[corpus_word_list_fixed['count']<=15]['count+1'], label='After +1 Smoothing')

plt.legend()
plt.title('Distribution of Word Counts Before/After \n Laplace +1 Smoothing')
plt.xlabel('Word Count Bin')
plt.ylabel('Count')
plt.show()
corpus.head()
corpus_2 = corpus[:int(np.round(len(corpus)*0.9,0))]
hold_out = corpus[int(np.round(len(corpus)*0.9,0))+1:]

corpus_2_list = " ".join(map(str, corpus_2))
hold_out_list = " ".join(map(str, hold_out))

# Test set word probabilities
hold_out_word_list = pd.DataFrame({'words':hold_out.str.split(' ', expand = True).stack().unique()})
hold_out_word_count_table = pd.DataFrame()
for n,word in enumerate(hold_out_word_list['words']):
    # Create a list of just the word we are interested in, we use regular expressions so that part of words do not count
    # e.g. 'ear' would be counted in each appearance of the word 'year'
    hold_out_word_count = len(re.findall(' ' + word + ' ', hold_out_list))  
    hold_out_word_count_table = hold_out_word_count_table.append(pd.DataFrame({'count':hold_out_word_count}, index=[n]))
    
    clear_output(wait=True)
    print('Proportion of words completed:', np.round(n/len(hold_out_word_list),4)*100,'%')

hold_out_word_list['count'] = hold_out_word_count_table['count']
# Remove the count for the start and end of sentence notation so 
# that these do not inflate the other probabilities
hold_out_word_list['count'] = np.where(hold_out_word_list['words'] == '<s' , 0,
                     np.where(hold_out_word_list['words'] == '/s>', 0,
                     hold_out_word_list['count']))
hold_out_word_list['prob'] = hold_out_word_list['count']/sum(hold_out_word_list['count'])


hold_out_Matrix = pd.DataFrame({'words': hold_out_word_list['words']})

start_time = time.time()


# Add limits to number of columns/rows so this doesn't run for ages
column_lim = 100
#column_lim = len(W_W_Matrix)
row_lim = 10
#row_lim = len(W_W_Matrix)

for r, column in enumerate(hold_out_Matrix['words'][0:column_lim]):
    
    prob_table = pd.DataFrame()
    for i, row in enumerate(hold_out_Matrix['words'][0:row_lim]):

        word_1 = ' ' + str(row) + ' '
        word_2 = str(column) + ' '

        if len(re.findall(word_1, hold_out_list)) == 0:
            prob = pd.DataFrame({'prob':[0]}, index=[i])
        else:
            prob = pd.DataFrame({'prob':[len(re.findall(word_1 + word_2, hold_out_list)) / len(re.findall(word_1, hold_out_list)) ]}, index=[i])
        
        prob_table = prob_table.append(prob)
    hold_out_Matrix[str(column)] = prob_table['prob']
    
    # Outputs progress of main loop, see:
    clear_output(wait=True)
    print('Proportion of column words completed:', np.round(r/len(hold_out_Matrix[0:column_lim]),2)*100,'%')
    
end_time = time.time()
print('Total run time = ', np.round(end_time-start_time,2)/60, ' minutes')

hold_out_Matrix.head(20)
hold_out_Matrix['words'] = hold_out_word_list['words']
lambda_1 = 0.5
lambda_2 = 0.5

# Create copy so we dont have to re-calculate original
hold_out_Matrix_2 = hold_out_Matrix.copy()
hold_out_Matrix_2 = hold_out_Matrix_2.dropna()
# Extract 'words' column 
hold_out_Matrix_3 = pd.DataFrame({'words':hold_out_Matrix_2.iloc[:,0]})
hold_out_Matrix_2 = hold_out_Matrix_2.iloc[:,1:]

# Multiply bigrams by lambda 1
hold_out_Matrix_2 = lambda_1*hold_out_Matrix_2

for n,column in enumerate(list(hold_out_Matrix_2)):
    column_prob = hold_out_word_list[hold_out_word_list['words']==column]['prob'].iloc[0]
    column_prob = lambda_2*column_prob
    
    hold_out_Matrix_3[str(column)] = hold_out_Matrix_2[column] + column_prob
    
    # Outputs progress of main loop, see:
    clear_output(wait=True)
    print('Proportion of column words completed:', np.round(n/len(list(hold_out_Matrix_2)),2)*100,'%')
    
# Sum probabilities of matrix (remove word column from calculation)
total_prob = hold_out_Matrix_3.iloc[:,1:].values.sum()

print(total_prob)
output_table = pd.DataFrame()

for x in range(0,11):
    lambda_1 = x/10
    lambda_2 = 1-lambda_1
        
    # Create copy so we dont have to re-calculate original
    hold_out_Matrix_2 = hold_out_Matrix.copy()
    hold_out_Matrix_2 = hold_out_Matrix_2.dropna()
    # Extract 'words' column 
    hold_out_Matrix_3 = pd.DataFrame({'words':hold_out_Matrix_2.iloc[:,0]})
    hold_out_Matrix_2 = hold_out_Matrix_2.iloc[:,1:]

    # Multiply bigrams by lambda 1
    hold_out_Matrix_2 = lambda_1*hold_out_Matrix_2

    for n,column in enumerate(list(hold_out_Matrix_2)):
        column_prob = hold_out_word_list[hold_out_word_list['words']==column]['prob'].iloc[0]
        column_prob = lambda_2*column_prob

        hold_out_Matrix_3[str(column)] = hold_out_Matrix_2[column] + column_prob

        # Outputs progress of main loop, see:
        clear_output(wait=True)
        print('Current lambda 1 value:', np.round(lambda_1,2))
        print('Current lambda 2 value:', np.round(lambda_2,2)) 
        print('Proportion of column words completed:', np.round(n/len(list(hold_out_Matrix_2)),2)*100,'%')


    # Sum probabilities of matrix (remove word column from calculation)
    total_prob = hold_out_Matrix_3.iloc[:,1:].values.sum()
    output_table = output_table.append(pd.DataFrame({'lambda_1':lambda_1,
                                                     'lambda_2':lambda_2,
                                                     'total_prob':total_prob}, index = [x]))
output_table.head()
output_table['lambda_1_2'] = 'L1:' + output_table['lambda_1'].astype(str) +' /L2: '+ output_table['lambda_2'].astype(str)
        
plt.bar(output_table['lambda_1_2'], output_table['total_prob'])
plt.title("Total Probability of Hold-out Set after \n Applying Simple Interpolation")
plt.xlabel('Lambda 1 and Lambda 2 Parameters')
plt.xticks(output_table['lambda_1_2'], rotation='vertical')
plt.xlabel('Total Probability')
plt.show()
optimal_lambda_1 = output_table[output_table['total_prob']==max(output_table['total_prob'])].iloc[0]['lambda_1']
optimal_lambda_2 = output_table[output_table['total_prob']==max(output_table['total_prob'])].iloc[0]['lambda_2']

print("Optimal Lambda 1 = ", optimal_lambda_1)
print("Optimal Lambda 2 = ", optimal_lambda_2)