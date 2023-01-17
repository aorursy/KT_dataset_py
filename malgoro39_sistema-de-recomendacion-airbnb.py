# Importando las Librerias

import pandas as pd

from IPython.display import Image, HTML

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel
# Importing the dataset

listings = pd.read_csv('../input/listings.csv', usecols = ['id', 'name', 'description'])

listings.head(10)
listings['name'] = listings['name'].astype('str')

listings['description'] = listings['description'].astype('str')
name_corpus = ' '.join(listings['name'])

description_corpus = ' '.join(listings['description'])
name_wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'white', height = 2000, width = 4000).generate(name_corpus)

plt.figure(figsize = (16,8))

plt.imshow(name_wordcloud)

plt.axis('off')

plt.show()
description_wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'white', height = 2000, width = 4000).generate(description_corpus)

plt.figure(figsize = (16,8))

plt.imshow(description_wordcloud)

plt.axis('off')

plt.show()
listings['content'] = listings[['name', 'description']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
# Fillna

listings['content'].fillna('Null', inplace = True)
tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')

tfidf_matrix = tf.fit_transform(listings['content'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}

for idx, row in listings.iterrows():

    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]

    similar_items = [(cosine_similarities[idx][i], listings['id'][i]) for i in similar_indices]

    results[row['id']] = similar_items[1:]
def item(id):

    name   = listings.loc[listings['id'] == id]['content'].tolist()[0].split(' // ')[0]

    desc   = ' \nDescription: ' + listings.loc[listings['id'] == id]['content'].tolist()[0].split(' // ')[1][0:165] + '...'

    prediction = name  + desc

    return prediction



def recommend(item_id, num):

    print('Recommending ' + str(num) + ' products similar to ' + item(item_id))

    print('---')

    recs = results[item_id][:num]

    for rec in recs:

        print('\nRecommended: ' + item(rec[1]) + '\n(score:' + str(rec[0]) + ')')
recommend(item_id = 4085439, num = 5)
recommend(item_id = 7021702, num = 5)