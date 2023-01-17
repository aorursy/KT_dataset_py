# Importing requests, BeautifulSoup and nltk

import requests

from bs4 import BeautifulSoup

import nltk

from nltk.stem import WordNetLemmatizer 

print('Done!')
import codecs



html = codecs.open("../input/2701-h.htm", 'r', 'utf-8')

print('Done!')
# Creating a BeautifulSoup object from the HTML

soup = BeautifulSoup(html)



# Getting the text out of the soup

text = soup.get_text()



# Printing out text between characters 32000 and 34000

print(text[32000:34000])
# Creating a tokenizer

tokenizer = nltk.tokenize.RegexpTokenizer('\w+')



# Tokenizing the text

tokens = tokenizer.tokenize(text)



# Printing out the first 8 words / tokens 

print(tokens[:8])
# A new list to hold the lowercased words

# Looping through the tokens and make them lower case

words = [word.lower() for word in tokens]



# Printing out the first 8 words / tokens 

print(words[:8])
# Getting the English stop words from nltk

sw = nltk.corpus.stopwords.words('english')



# Printing out the first eight stop words

print(sw[:8])
# A new list to hold Moby Dick with No Stop words

# Appending to words_ns all words that are in words but not in sw



words_ns = [word for word in words if word not in sw]



# Printing the first 5 words_ns to check that stop words are gone

print(words_ns[:10])
lemmatizer = WordNetLemmatizer() 



print("rocks :", lemmatizer.lemmatize("rocks")) 

print("corpora :", lemmatizer.lemmatize("corpora")) 

  

# a denotes adjective in "pos" 

print("better :", lemmatizer.lemmatize("better", pos ="a")) 
words_lem = [lemmatizer.lemmatize(word) for word in words_ns]



print(words_lem[:10])
# This command display figures inline

from matplotlib.pyplot import figure

%matplotlib inline



# Creating the word frequency distribution

freqdist = nltk.FreqDist(words_ns)



# Plotting the word frequency distribution

figure(figsize=(10,5))

freqdist.plot(25)
# What's the most common word in Moby Dick?

most_common_word = 'whale'