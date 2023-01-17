import pandas as pd
articles_df = pd.read_csv('../input/shared_articles.csv')

articles_df.head(5)
#import logging

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



from gensim import corpora, models, similarities

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords
#Filtering only English articles

english_articles_df = articles_df[articles_df['lang'] == 'en']

#Concatenating the articles titles and bodies

english_articles_content = (english_articles_df['title'] + ' ' + english_articles_df['text']).tolist()
#Loading a set of English stopwords

english_stopset = set(stopwords.words('english')).union(

                  {"things", "that's", "something", "take", "don't", "may", "want", "you're", 

                   "set", "might", "says", "including", "lot", "much", "said", "know", 

                   "good", "step", "often", "going", "thing", "things", "think",

                  "back", "actually", "better", "look", "find", "right", "example", 

                   "verb", "verbs"})
#Tokenizing words of articles

tokenizer = RegexpTokenizer(r"(?u)[\b\#a-zA-Z][\w&-_]+\b")

english_articles_tokens = list(map(lambda d: [token for token in tokenizer.tokenize(d.lower()) if token not in english_stopset], english_articles_content))
#Processing bigrams from unigrams (sets of two works frequently together in the corpus)

bigram_transformer = models.Phrases(english_articles_tokens)

english_articles_unigrams_bigrams_tokens = list(bigram_transformer[english_articles_tokens])
#Creating a dictionary and filtering out too rare and too common tokens

english_dictionary = corpora.Dictionary(english_articles_unigrams_bigrams_tokens)

english_dictionary.filter_extremes(no_below=5, no_above=0.4, keep_n=None)

english_dictionary.compactify()

print(english_dictionary)
#Processing Bag-of-Words (BoW) for each article

english_articles_bow = [english_dictionary.doc2bow(doc) for doc in english_articles_unigrams_bigrams_tokens]
#Training the LDA topic model on English articles

lda_model = models.LdaModel(english_articles_bow, id2word=english_dictionary, num_topics=30, passes=10, iterations=500)
#Processing the topics for each article

english_articles_lda = lda_model[english_articles_bow]
def get_topics_top_words(model, max_words):

    all_topics = model.show_topics(-1, max_words*2, False, False)

    topics = []

    for topic in all_topics:    

        min_score_word = float(abs(topic[1][0][1])) / 2.

        top_positive_words = list(map(lambda y: y[0].replace('_',' '), filter(lambda x: x[1] > min_score_word, topic[1])))[0:max_words]

        topics.append('[' + ', '.join(top_positive_words) + ']')

    return topics



#Computing the main topic of each article

topics_top_words = get_topics_top_words(lda_model, 5)
def get_main_topics(corpus_lda, topics_labels):

    min_strength = (1.0 / float(len(topics_labels))) + 0.01

    main_topics = map(lambda ts: sorted(ts, key=lambda t: -t[1])[0][0] if sorted(ts, key=lambda t: -t[1])[0][1] > min_strength else None, corpus_lda)

    main_topics_labels = map(lambda x: topics_labels[x] if x != None else '', main_topics)

    return list(main_topics_labels)



#Return the discovered topics, sorted by popularity

corpus_main_topics = get_main_topics(english_articles_lda, topics_top_words)



main_topics_df = pd.DataFrame(corpus_main_topics, columns=['topic']).groupby('topic').size().sort_values(ascending=True).reset_index()

main_topics_df.columns = ['topic','count']

main_topics_df.sort_values('count', ascending=False)
main_topics_df.plot(kind='barh', x='topic', y='count', figsize=(7,20), title='Main topics on shared English articles')