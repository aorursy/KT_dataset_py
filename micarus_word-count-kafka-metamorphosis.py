#the necessary imports
from bs4 import BeautifulSoup
import requests
import nltk
#this gets the book from a certain website
r = requests.get('http://www.gutenberg.org/files/5200/5200-h/5200-h.htm')

#whenever you have a book you need to set the encoding correctly, most of the cases this is 'utf-8'
r.encoding = 'utf-8'

# Now lets extract the text from the html which is placed in our variable r
html = r.text

# Lets do a sanity and check the first 1000 words
print(html[:1000])
# Lets first create the soup from our HTML file
soup = BeautifulSoup(html)

# Then we're getting the text out of it
text = soup.get_text()

# Lets print a random area to make sure that everything is working fine
print(text[10000:11000])
# First we create the tokenizer (\w+) means all non-word characters
tokenizer = nltk.tokenize.RegexpTokenizer('\w+')

# Then we will fill in the text
tokens = tokenizer.tokenize(text)

# Lets do a sanity check again
print(tokens[:4])
# Lets make a new list with lowercase words
words = []

# Looping through the words and appending them in the new list.
for word in tokens:
    words.append(word.lower())

# Sanity, sanity, sanity
print(words[:4])
#For the stopwords we have to install nltk.corpus, else you will get an error. Lets place the stopwords in sw
from nltk.corpus import stopwords
sw = stopwords.words('english')

# We create a new list without any stopword, called words_ns
words_ns = []

# Appending to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)

# Lets make sure that the stopwords are gone, as you can see 'by' is now gone
print(words_ns[:4])
# Displays figures inline in Jupyter Notebooks
%matplotlib inline

# Create a frequency distribution
freqdist = nltk.FreqDist(words_ns)

# Lets plot and see if it has paid off!
freqdist.plot


# The word found most is 'Gregor'