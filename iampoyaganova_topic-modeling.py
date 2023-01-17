import numpy as np 

import pandas as pd 
train = pd.read_csv('../input/train.csv')
train = train[:10000]
import spacy

import re

from nltk.corpus import stopwords

nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

stops = stopwords.words("english")

def normalize(comm, lowercase, remove_stopwords):

    text = re.sub(r'(\w)\1{2,}', r'\1\1', comm)

    comment = re.sub(r'(\W)\1+', r'\1', text)

    if lowercase:

        comment = comment.lower()

    comment = nlp(comment)

    lemmatized = list()

    for word in comment:

        lemma = word.lemma_.strip()

        if lemma:

            if not remove_stopwords or (remove_stopwords and lemma not in stops):

                lemmatized.append(lemma)

    docs = nlp.pipe(lemmatized, batch_size=1000, n_threads=4)

    return [' '.join([x.lemma_ for x in doc if x.is_alpha]) for doc in docs]
x_train_lemmatized = train.comment_text.apply(normalize, lowercase=True, remove_stopwords=True)
train['comment_text'] = x_train_lemmatized

train.head()
def reg(corpus):

    corpus1 = []

    for i in corpus:

        corpus1.append(' '.join(i))

    return corpus1

train['comment_text'] = reg(train.comment_text)

train.head()
toxics = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

x_train, x_test, y_train, y_test = train_test_split(train, train[toxics], test_size=0.3, shuffle=True)

x_train.head()
# задаем векторизатор tf-idf



from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_df=0.87,

               smooth_idf=True, use_idf=True, max_features=300000)
from sklearn.decomposition import TruncatedSVD

from sklearn.pipeline import Pipeline
# применяем SVD для снижения размерности  



svd_model = TruncatedSVD(n_components=3,

                         algorithm='randomized',

                         n_iter=10)



# pipeline: tf-idf + SVD

svd_transformer = Pipeline([('tfidf', vect), 

                            ('svd', svd_model)])



svd_train = svd_transformer.fit_transform(x_train['comment_text'])

svd_test = svd_transformer.transform(x_test['comment_text'])



# получаем svd матрицу для дальнейшей загрузки в модель 
# используем логистическую регрессию в качестве классификатора 



from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression



submiss_1 = x_test['id']

logreg = LogisticRegression(C=12, max_iter=10000, dual=True)



submiss_1 = pd.DataFrame.from_dict({'id': x_test['id']})

for i in toxics:

    logreg.fit(svd_train, y_train[i])

    

    submiss_1.loc[:,i] = logreg.predict_proba(svd_test)[:,1]

submiss_1.head()
#делаем вывод, что на нашей выборке методика tf-idf + tSVD в качестве латентно-семантического анализа не является эффективной 

#например, обычный tf-idf + LogReg дает лучшее качество



scores = []

for i in toxics:

    scores.append(roc_auc_score(y_test[i], submiss_1[i]))

np.mean(scores)
new = []

for i in train['comment_text']:

    new.append(i.split())
from gensim import corpora

dictionary = corpora.Dictionary(new)

corpus = [dictionary.doc2bow(text) for text in new]

import pickle

pickle.dump(corpus, open('corpus.pkl', 'wb'))

dictionary.save('dictionary.gensim')
# задаем число тем:  7 



import gensim

NUM_TOPICS = 7

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)

ldamodel.save('model7.gensim')

topics = ldamodel.print_topics(num_words=4)

for topic in topics:

    print(topic)
# визуализируем 7 тем на нашей коллекции комментариев 



dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')

corpus = pickle.load(open('corpus.pkl', 'rb'))

lda = gensim.models.ldamodel.LdaModel.load('model7.gensim')

import pyLDAvis.gensim

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)