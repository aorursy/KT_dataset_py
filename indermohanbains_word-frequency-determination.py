# Importing requests, BeautifulSoup and nltk
import requests
from bs4 import BeautifulSoup
import nltk
import matplotlib.pyplot as plt
import pandas as pd
# Getting the Moby Dick HTML 
r = requests.get ('https://s3.amazonaws.com/assets.datacamp.com/production/project_147/datasets/2701-h.htm')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print (html [0:2000])
# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup (html)

# Getting the text out of the soup
text = soup.get_text ()

# Printing out text between characters 32000 and 34000
print (text [32000 : 34000])
# Creating a tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer ('\w+')

# Tokenizing the text
tokens = tokenizer.tokenize (text)

# Printing out the first 8 words / tokens 
print (tokens [0:8])
# A new list to hold the lowercased words
words = []

# Looping through the tokens and make them lower case
for word in tokens:
    words.append (word.lower ())
# Printing out the first 8 words / tokens 
print (words [0:8])
# Getting the English stop words from nltk
from nltk.corpus import stopwords
sw = stopwords.words('english')

# Printing out the first eight stop words
print (sw [0:8])
# A new list to hold Moby Dick with No Stop words
words_ns = []

# Appending to words_ns all words that are in words but not in sw
for word in words:
    
    if word not in sw:
        words_ns.append (word)

# Printing the first 5 words_ns to check that stop words are gone
print (words_ns [0:5])
df = pd.DataFrame (words_ns)
df = df.reset_index ()
df.columns = ['No', 'words']
W = df ['words'].value_counts ().head (25)
W.to_dict ()
# This command display figures inline
%matplotlib inline

# Creating the word frequency distribution
freqdist = nltk.FreqDist (words_ns)

# Plotting the word frequency distribution
freqdist.plot (25)
# What's the most common word in Moby Dick?
most_common_word = 'whale'
