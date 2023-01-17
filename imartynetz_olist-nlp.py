import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
order_review_data = pd.read_csv("../input/olist_order_reviews_dataset.csv")
order_review_data.head()
order_review_data.info()
order_review_data = order_review_data.dropna(subset=['review_comment_message'])
order_review_data['word_count'] = order_review_data.review_comment_message.apply(lambda x: len(str(x).split()))
order_review_data.word_count.max()
g = sns.FacetGrid(data=order_review_data, col='review_score',height=5, aspect=0.8)
before_remove = g.map(plt.hist, 'word_count', bins=30)
before_remove
sns.boxplot(x='review_score', y='word_count', data=order_review_data)
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import RSLPStemmer #Stemmer for portugese words.

from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest

stop = stopwords.words('portuguese')
stop.append('nao') #Stopword already have "Não", just adding this because it's appear on dataframe
text_review_1 = ' '.join(order_review_data[order_review_data["review_score"]==1]["review_comment_message"])
text_review_2 = ' '.join(order_review_data[order_review_data["review_score"]==2]["review_comment_message"])
text_review_3 = ' '.join(order_review_data[order_review_data["review_score"]==3]["review_comment_message"])
text_review_4 = ' '.join(order_review_data[order_review_data["review_score"]==4]["review_comment_message"])
text_review_5 = ' '.join(order_review_data[order_review_data["review_score"]==5]["review_comment_message"])
def resumo (texto,n):
    sentencas = sent_tokenize(texto)
    palavras = word_tokenize(texto.lower())
    
    stop = set(stopwords.words('portuguese') + list(punctuation))
    palavras_sem_stopwords = [palavra for palavra in palavras if palavra not in stop]
    
    frequencia = FreqDist(palavras_sem_stopwords)
    sentencas_importantes = defaultdict(int)
    
    for i, sentenca in enumerate(sentencas):
        for palavra in word_tokenize(sentenca.lower()):
            if palavra in frequencia:
                sentencas_importantes[i] += frequencia[palavra]
                
    idx_sentencas_importantes = nlargest(n, sentencas_importantes, sentencas_importantes.get)
    for i in sorted(idx_sentencas_importantes):
        print(sentencas[i])        
def visualize(label):
    words = ''
    for msg in order_review_data[order_review_data['review_score'] == label]['review_comment_message']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
visualize(1)
resumo(text_review_1,4)
visualize(2)
resumo(text_review_2,4)
visualize(3)
resumo(text_review_3,4)
visualize(4)
resumo(text_review_4,4)
visualize(5)
resumo(text_review_5,4)
stemmer = RSLPStemmer()
import re
import unicodedata

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)
#removing numbers
order_review_data.review_comment_message = order_review_data.review_comment_message.str.replace('\d+', ' ')

#lower cases.
order_review_data.review_comment_message = order_review_data.review_comment_message.apply(lambda x: " ".join(x.lower() for x in x.split()))

#Removing punctuation
order_review_data.review_comment_message = order_review_data.review_comment_message.str.replace('[^\w\s]',' ')

#Removing stopword
order_review_data.review_comment_message = order_review_data.review_comment_message.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#removing accentuation
order_review_data.review_comment_message = order_review_data.review_comment_message.apply(strip_accents)

#Tokenmize
order_review_data.review_comment_message = order_review_data.apply(lambda row: word_tokenize(row['review_comment_message']), axis=1)

#Stemming
order_review_data.review_comment_message = order_review_data.review_comment_message.apply(lambda x: " ".join([stemmer.stem(word) for word in x]))
order_review_data['word_count_new'] = order_review_data.review_comment_message.apply(lambda x: len(str(x).split()))
order_review_data.head()
order_review_data.word_count_new.max()
g = sns.FacetGrid(data=order_review_data, col='review_score',height=5, aspect=0.8)
g.map(plt.hist, 'word_count_new', bins=30)
sns.boxplot(x='review_score', y='word_count_new', data=order_review_data)
order_training = order_review_data
order_training['review_score'][order_training.review_score == 2] = 1
order_training['review_score'][order_training.review_score == 3] = 1
order_training['review_score'][order_training.review_score == 4] = 5
order_training = order_training[(order_training.review_score == 1) | (order_training.review_score == 5)]
order_training = order_training.loc[:,['review_comment_message','review_score']]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_count = cv.fit_transform(order_training["review_comment_message"]).toarray()
y_count = order_training.review_score
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_count_pca = pca.fit_transform(X_count)
pca.explained_variance_ratio_
plt.figure(figsize=(12,8))
sns.scatterplot(x = X_count_pca[:,0], y = X_count_pca[:,1] , hue=y_count,palette = 'RdYlBu', legend="full")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_count, y_count, test_size = 1/4, random_state = 42)
logreg_vector = LogisticRegression()
logreg_vector.fit(X_train, y_train)
y_pred = logreg_vector.predict(X_test)
score = logreg_vector.score(X_test, y_test)
print(score)
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)
def plot_coefficients(classifier, feature_names, top_features=20):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(20, 8))
    colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors, align="center")
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=90)
    plt.xlabel("20 more significant words for bad reviews (red) and good reviews (right)")
    plt.ylabel("Coeficient values")
    
plot_coefficients(logreg_vector, cv.get_feature_names())
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(order_training.review_comment_message)
X_tfidf = X_tfidf.todense()
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_tfidf_pca = pca.fit_transform(X_tfidf)
pca.explained_variance_ratio_
plt.figure(figsize=(12,8))
sns.scatterplot(x = X_tfidf_pca[:,0], y = X_tfidf_pca[:,1] , hue=y_count, palette = 'RdYlBu', legend="full")
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y_count, test_size = 1/4, random_state = 42)
logreg_tfidf = LogisticRegression()
logreg_tfidf.fit(X_train_tfidf, y_train_tfidf)
y_pred_tfidf = logreg_tfidf.predict(X_test_tfidf)
score = logreg_tfidf.score(X_test_tfidf, y_test_tfidf)
print(score)
cm = metrics.confusion_matrix(y_test_tfidf, y_pred_tfidf)
print(cm)
plot_coefficients(logreg_tfidf, vectorizer.get_feature_names())