import pandas as pd
import re

from nltk.tokenize import RegexpTokenizer
import nltk

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/titles.txt', sep='\n')
df.head()
df.columns = ['articles']
df.info()
# starting with duplicates drop to avoid unnecessary processing
df.drop_duplicates(inplace=True)
df.info()
# changing all row values to strings
df.articles = df.articles.apply(str)
# using lower on each row to remove values differing only by case
df.articles = [article.lower() for article in df.articles]
# changing data frame to a list, it will allow quick list comprehensions
articles = df.articles.tolist()
articles = list(set(articles))
len(articles)
# changing underscores to spaces
articles = [re.sub('_', ' ', article) for article in articles]
# removing special-characters-only rows
articles = [re.sub('[^A-Za-z0-9\s]+', '', article) for article in articles]
articles[:10]
# removing empty strings left
final_articles = [article for article in articles if article != '']
# removing unnecessary spaces
final_articles = [article.strip() for article in final_articles]
final_articles[:10]
len(final_articles)
# splitting each article so that in next steps I can create one string of all the text
all_lines = [article.split(' ') for article in final_articles]
all_lines[:10]
all_words = ' '.join(word for line in all_lines for word in line if word != '')
all_words[:1]
# tokenizing words
tokenizer = RegexpTokenizer('\w+')
tokens = tokenizer.tokenize(all_words)
print(tokens[:5])
# creating freq dist and plot
freqdist1 = nltk.FreqDist(tokens)
freqdist1.plot(25)
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')
sw[:5]
all_no_stopwords = []
for word in tokens:
    if word not in sw:
        all_no_stopwords.append(word)

freqdist2 = nltk.FreqDist(all_no_stopwords)
freqdist2.plot(25)
freqdist2.most_common(25)
def contextWords(search_word, list_of_strings):
    '''
    Takes a string as an argument and checks it's presence in each string from a given list.
    Returns a frequency distribution plot of words that accompany the search word in all strings, excluding English 
    stopwords.
    '''
    search_list = [word for word in list_of_strings if re.findall(search_word, word)]
    search_words = []
    for expression in search_list:
        for w in expression.split(' '):
            if w not in nltk.corpus.stopwords.words('english'):
                search_words.append(w)
    new_search_words = ' '.join(word for word in search_words if word != '')
    new_tokens = tokenizer.tokenize(new_search_words)
    search_freqdist = nltk.FreqDist(new_tokens)
    search_freqdist.plot(20)   
contextWords('poland', final_articles)
contextWords('polska', final_articles)
contextWords('germany', final_articles)
contextWords('deutschland', final_articles)
