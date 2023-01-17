#this is a data manipulation library centered around the Data Frame, which stores two-dimensional data in a tabular format 
#and makes certain operations easier to carry out
import pandas as pd

#these are different elements from NLTK (Natural Language Toolkit), a massive natural language processing library
#  we are importing individual elements: it's much cleaner to onlly bring in what we need
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk.collocations
from nltk import bigrams

#these are utility libraries; they contain useful data structures, collections, or functions 
#that will make our code more simple
from string import punctuation
from collections import Counter
import glob
import csv

#these are a couple of plotting libraries, which we'll use at the very end of the exercise
import matplotlib.pyplot as plt
import seaborn as sns
#a wrapper function that turns frequencies into counts
#  don't worry about this!
def get_bigram_count(blog_tokens, n = 5):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(blog_tokens)
    bigram_finder.apply_freq_filter(3)

    freq_n = bigram_finder.score_ngrams(bigram_measures.raw_freq)[0 : n]

    num_bigrams = len([x for x in nltk.bigrams(blog_tokens)])

    for i in range(len(freq_n)):
        new_tup = (freq_n[i][0], round(freq_n[i][1] * num_bigrams))
        freq_n[i] = new_tup

    return freq_n
#these are additional stopwords for use later... don't worry about this right now!
ADD_STOPS = [p for p in punctuation] + ["''", '""', "``", '`', '...', '’', "'s", "n't"]
#checking that all of the desired files exist in the environment and are detected by Kaggle
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#get all the text files in the directory
#  we're looking for .txt files, but we don't care what their name is so we use "*" as a sort of "wildcard"
file_names = glob.glob(STRING) #TODO: FILL IN THE STRING
#taking a look at the files that we matched with our pattern
#  these are filepaths; they are the instructions for arriving at a certain file, interpreted by the computer
#TODO: ADD A PRINT STATEMENT
#creating a list to read all of the text into
file_strs = []

#going one by one, inserting the text from each file into the list as a string
for fp in file_names:
    with open(fp, 'r') as file:
        pass #TODO: REPLACE PASS WITH A STATEMENT THAT READS IN THE FILES
        
#how many files did we get?        
#TODO: ADD A PRINT STATEMENT
#taking a look at one of the blogs in our list
#  in "file_strs[0][0 : 500]", "[0]" gets the first blog and the "[0 : 500]" gets the first 500 characters
#TODO: PRINT OUT THE FIRST 500 CHARACTERS OF THE FIRST BLOG
#putting all of the files together into one string
#TODO: PUT ALL BLOGS INTO ONE STRING
#notice how tokens can be punctuation and that words like "don't" get split into "do" and "n't"
string_a = 'Hello there, how are you today? I am doing fine. Don\'t get caught in the rain!'
print(word_tokenize(STRING)) #TODO: REPLACE "STRING" WITH THE APPROPRIATE VARIABLE NAME
string_b = 'Hello'
string_c = 'hello'
print(string_b == string_c) #they're considered different
print(string_b.lower() == string_c.lower()) #now they're the same!
#taking a look at the first few stopwords from NLTK's stopword list
stopwords.words('english')[BOUND_A : BOUND_B] #TODO: REPLACE THE "BOUND_A" AND "BOUND_B" WITH THE CORRECT INTEGER BOUNDS
def preprocess_text(text_str, stopwords):
    str_lower = text_str.METHOD() #TODO: REPLACE "METHOD" WITH THE CORRECT STRING METHOD
    str_tokens = FUNCTION(str_lower) #TODO: REPLACE FUNCTION WITH THE FUNCTION FOR TOKENIZATION
    
    final_tokens = [] #the list to hold non-stopword tokens
    for token in str_tokens:
        pass #TODO: REPLACE "PASS" WITH THE CONTENTS OF THE LOOP - ONLY KEEP NON-STOPWORDS
            
    return VARIABLE #TODO: REPLACE "VARIABLE" WITH THE CORRECT VARIABLE TO RETURN
#normalize case, tokenize, and remove stopwords/punctuation
stops = stopwords.words('english') + ADD_STOPS #our ADD_STOPS stopwords include punctuation and a little bit more

#a list to hold all of our tokens for each text file
each_blog_tokens = OBJECT #TODO: REPLACE "OBJECT" WITH CORRECT OBJECT TO HOLD OUR TOKENS

for i in range(len(file_strs)):
    processed_blog = FUNCTION() #TODO: APPLY THE FUNCTION THAT WE CREATED - REPLACE "FUNCTION()"
    
    each_blog_tokens.append(VARIABLE) #TODO: REPLACE "VARIABLE" WITH THE VARIABLE THAT WE WANT TO ADD TO THE LIST
#seeing how we did with tokenization
print(each_blog_tokens[0][0 : 50])
#"all_blog_tokens" stores each blogs' tokens in one list whereas "each_blog_tokens" stores each blogs' tokens in seperate lists
all_blog_tokens = preprocess_text(all_blogs, stops)

print(len(all_blog_tokens)) #the total amount of tokens in all blogs
#looking at the most frequent unigrams
most_common = []

#getting the ten most frequent for each text
for blog in each_blog_tokens:
    freq_dict = OBJECT(blog).METHOD(10) #TODO: REPLACE "OBJECT" AND "METHOD" 
    most_common.append(freq_dict) #adding it to our list
#seeing what the result is for the first blog post is
print(most_common[INTEGER]) #TODO: REPLACE "INTEGER" WITH THE CORRECT INDEX
#getting the ten most frequent unigrams across ALL texts
freq_dict_all = Counter(all_blog_tokens).most_common(10)

print(freq_dict_all)
#looking at the most frequent and most associated bigrams
bigram_measures = nltk.collocations.BigramAssocMeasures() #these will help us score the bigrams

most_associated = []
most_freq = []

#extracting and scoring the bigrams for all texts
for blog in each_blog_tokens:
    bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(blog) #building a bigram finder 
    bigram_finder.apply_freq_filter(INTEGER) #TODO: REPLACE "INTEGER" WITH THE A THRESHOLD FOR FREQUENCY

    #scoring bigrams by PMI
    assoc_five = bigram_finder.score_ngrams(bigram_measures.pmi)[0 : 5]
    for i in range(len(assoc_five)): #rounding each bigram's association to three decimal places
        assoc_five[i] = (assoc_five[i][0], round(assoc_five[i][1], 3))
    OBJECT.append(assoc_five) #TODO: REPLACE "OBJECT" WITH THE LIST THAT WE WANT TO ADD TO
    
    #scoring bigrams by raw count
    freq_five = get_bigram_count(blog)
    OBJECT.append(freq_five) #TODO: REPLACE "OBJECT" WITH THE LIST THAT WE WANT TO ADD TO
#looking at the result for the most frequent bigrams in the first blog post
print(LIST[INDEX]) #TODO: REPLACE "LIST[INDEX]" WITH THE CORRECT LIST AND INTEGER INDEX
#looking at the result for the most associated bigrams in the first blog post, by PMI
print(LIST[INDEX]) #TODO: REPLACE "LIST[INDEX]" WITH THE CORRECT LIST AND INTEGER INDEX
#the header for each file
header_assoc = ['bigram', 'PMI_score', 'blog_name']
header_freq = ['bigram', 'count', 'blog_name']

#the rows for each file, which must still be populated
rows_assoc = []
rows_freq = []

ct = 0 #keeping count of where we are within the individual blogs
for blog in most_associated:
    for bigram in blog: #looking at each of the most associated bigrams
        bigram_str = bigram[0][0] + '/' + bigram[0][1] #building a string to represent the bigram
        blog_name = file_names[ct].split('/')[-1][0 : -4] #adding in the blog name as included in the original filepath
        rows_assoc.append([bigram_str, bigram[1], blog_name]) #adding in each element 
    ct += 1
    
#the same process as before, but for the most frequent bigrams
ct = 0
for blog in most_freq:
    for bigram in blog:
        bigram_str = bigram[0][0] + '/' + bigram[0][1]
        blog_name = file_names[ct].split('/')[-1][0 : -4]
        rows_freq.append([bigram_str, bigram[1], blog_name])
    ct += 1
#writing to the files
with open('most_associated.csv', 'w') as file: #opening a connection
    csvwriter = csv.writer(file) #building a csvwriter object to format our header and rows correctly
    
    csvwriter.writerow(header_assoc) #adding the header
    csvwriter.writerows(rows_assoc) #adding all of the rows
    
#the same proces as before, but for the most frequent bigrams    
with open('most_frequent.csv', 'w') as file:
    csvwriter = csv.writer(file)
    
    csvwriter.writerow(header_freq)
    csvwriter.writerows(rows_freq)
most_freq_df = pd.read_csv(FILEPATH) #TODO: ADD FILEPATH TO THE DATA

#dataframes are displayed nicely by Kaggle as long as you don't use "print()"!
most_freq_df.METHOD() #TODO: REPLACE "METHOD" WITH THE METHOD THAT SHOWS THE FIRST FIVE OBSERVATIONS
#grabbing just the observations that correspond to the blog "Acephalous-Internet"
aceph = most_freq_df[most_freq_df['blog_name'] == STRING] #TODO: REPLACE "STRING" WITH THE DESIRED BLOG NAME

OBJECT #TODO: REPLACE "OBJECT" WITH THE CORRECT DATA FRAME
#making a barplot using seaborn - we could do this for any of the blogs!
sns.barplot(data = aceph, y = STRING_1, x = STRING_2) #TODO: REPLACE "STRING_1" AND "STRING_2" WITH THE CORRECT COLUMN NAMES

plt.savefig(FILENAME, bbox_inches = 'tight', dpi = 400) #TODO: REPLACE "FILENAME" WITH THE DESIRED FILE NAME