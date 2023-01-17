# PROBLEM STATMENT 1 | LANGUAGE MODELING
# This is a kaggle kernel notebook using 
# DATASET SOURCE: Amazon Fine Food Review dataset: https://www.kaggle.com/snap/amazon-fine-food-reviews
# I have made this notebook public for authentication purpose here: 

import pandas as pd # just to input and output CSV file I/O (e.g. pd.read_csv)
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
data.head()
data = data[['Text']][:1000000]
data.head()
print("We will use only top 5000 reviews to prcess out model", data.shape)
data['Text'][2]
sentences = []
for review in data['Text']:
    # split sentences
    x = re.split(r' *[\.\?!][\'"\)\]]* *', review)     
    
    # marking <eos> as full stop
    x = ' . '.join(x)
    
    #collection of these clean sentences
    sentences.append(x)
    
# create a df column
data['separate_sentences'] = sentences
data.head()

# Reference: https://stackoverflow.com/a/47091490/4084039
# replace slangs with full words
# e.g. replace [ can't   ->   can not ]

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase
# common function to pre process whole data

preprocessed_reviews = []

for sentance in data['separate_sentences'].values:
    
    #remove url
    sentance = re.sub(r"http\S+", "", sentance)
    
    #remove tags 
    sentance = re.sub('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', '', sentance)
    
    #convert short forms into full forms
    sentance = decontracted(sentance)
    
    #remove words with numbers
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    
    #remove spacial character EXCEPT <eos> i.e. fullstop (.):
    sentance = re.sub('[^A-Za-z.]+', ' ', sentance)
    
    sentance = ' '.join(e.lower() for e in sentance.split())
    preprocessed_reviews.append(sentance.strip())
print("Original Text: \n\n", data['Text'][2])
print("-"*50)
print("In sentence format: \n\n", data['separate_sentences'][2])
print("-"*50)
print("Preprocessed: \n\n",preprocessed_reviews[2])
data['separate_sentences'] = preprocessed_reviews
data.head()
# dictionary to catch frequency of each word or say create a bag of words
bow = {}

for sentence in data['separate_sentences']:
    temp = sentence.split()
    for token in range(len(temp[:-1])):
        if temp[token]+' '+temp[token+1] in bow.keys():
            bow[str(temp[token]+' '+temp[token+1])]+=1
        else:
            bow[str(temp[token]+' '+temp[token+1])]=1
            
print(" \"FEW\" BiGram BoW showing frequencies of bigrams : \n\n")
temp=0
for key,value in bow.items():
    print(key, ":", value)
    temp+=1
    if temp >30:
        break
# dictionary to catch frequency of each word or say create a bag of words
bow_unigram = {}

for sentence in data['separate_sentences']:
    temp = sentence.split()
    for token in temp:
        if token in bow_unigram.keys():
            bow_unigram[token]+=1
        else:
            bow_unigram[token]=1


bow_unigram['.'] = 0
print(" \"FEW\" UniGram BoW showing frequencies of unigrams : \n\n")
temp=0
for key,value in bow_unigram.items():
    print(key, ":", value)
    temp+=1
    if temp >30:
        break
#Probability Formulae

def prob(w_dash,w):    
    #variable to store COUNT(W,W')
    count_bigram = 0
    #check occurence of bigram
    bigram = str(str(w.strip())+' '+str(w_dash.strip()))
    if bigram in bow.keys():
        count_bigram = bow[bigram]
    
    #variable to store unigram count
    count_uni = 0
    #check ocurences of unigram
    unigram = str(w.strip())
    if unigram in bow_unigram.keys():
        count_uni = bow_unigram[unigram]
       
    #make sure denominator is not zero
    if count_uni:
        #return probability as mentioned in problem statement formulae.
        return (count_bigram/count_uni)
        #ALSO WHILE IMPLEMENTING NAIVE BAYES, ALPHA (LAPLACE SMOOTHING) IS DONE TO REMOVE THE PROBLEM OF DIVISION BY 0.
        # This could be understood by the practical case as when 
        # "IN TEST DATA" a word is encountered which has not been appeared yet in train data then division by 0 problem will occur
    return 0 
#function to find next word by comparing all the possible bigrams and occurences and fethcing the bigram with maximum probability

def find_next(word):
    l = [(key,bow[key]) for key in bow.keys() if word.lower()+' ' in key]
    #var to store max probability for the w_dash being the new word. 
    max_prob = 0
    #var to store next word
    next_word = None
    
    for each in l:        
        
        w, w_dash = each[0].split()
        
        #finding probability
        p = prob(w_dash, w)
        
        #only update next word if the new probability is more than the max prob
        if p>max_prob:
            max_prob = p
            next_word = w_dash
    
    #return next word
    return next_word
#take first word as input from user
a = input()
#predicted sentence is formed in this variable
predicted_sentence = ""+a
next_word = a

#maximum length is set to 10 but if a fullstop is encountered then it will break in mid way...
while(len(predicted_sentence.split())<10):
    next_word = find_next(next_word)
    predicted_sentence=predicted_sentence+' '+next_word.strip()
    print(predicted_sentence)
    
    #break if <eos> is encountered
    if next_word.strip() is '.':
        break
    