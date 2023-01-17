



import pandas as pd
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


# changing data frame to a list, it will allow quick list comprehensions
articles = df.articles.tolist()
articles = list(set(articles))
len(articles)
articles[0:100]

import re
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

final_articles[50:100]
corpus=[]
for title in final_articles:
    corpus.append(title.lower().split())
print("corpusun alınması işlemi başladı")    
from gensim.models import Word2Vec
model=Word2Vec(corpus,size=100, window=5, min_count=3)


model['book']
model.wv.most_similar("book")
model.wv.most_similar(positive=["king","woman"],negative=["man"])



# tokenizing words
# creating freq dist and plot








