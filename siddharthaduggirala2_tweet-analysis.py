# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd 
import numpy as np 
import nltk
import matplotlib.pyplot as plt 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from wordcloud import WordCloud


data_train = pd.read_csv('../input/train.csv', encoding='latin-1')


def remove_pattern(text, pattern):
    
    r = re.findall(pattern, text)
    
    for i in r:
        
        text = re.sub(i, '',text)
        
    return text

data_train['clean_tweet'] = np.vectorize(remove_pattern)(data_train['SentimentText'], "@[\w]*")

data_train['clean_tweet'] = data_train['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

data_train['clean_tweet'] = data_train['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


tokenzied_tweet = data_train['clean_tweet'].apply(lambda x: x.split())


stemmer = PorterStemmer()

tokenzied_tweet = tokenzied_tweet.apply(lambda x: [stemmer.stem(i) for i in x])


for i in range(len(tokenzied_tweet)):
    
    tokenzied_tweet[i] = ' '.join(tokenzied_tweet[i])
    
data_train['clean_tweet'] = tokenzied_tweet

summary = data_train['clean_tweet']
score = data_train['Sentiment']

# word cloud of all words
all_words = ' '.join([text for text in data_train['clean_tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



# negative words wordcloud

negative_words = ' '.join([text for text in data_train['clean_tweet'][data_train['Sentiment'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

positive_words =' '.join([text for text in data_train['clean_tweet'][data_train['Sentiment'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(summary, score, test_size=0.2, random_state=42)

count_vect = CountVectorizer()

X_trains_counts = count_vect.fit_transform(X_train)


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_trains_counts)


X_new_counts = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_new_counts)


prediction = dict()

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, y_train)
prediction['Multinomial'] = model.predict(X_test_tfidf)

from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X_train_tfidf, y_train)
prediction['Bernoulli'] = model.predict(X_test_tfidf)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(X_train_tfidf, y_train)
prediction['Logistic'] = logreg.predict(X_test_tfidf)


cmp = 0
colors = ['b', 'g', 'y', 'm', 'k']
for model, predicted in prediction.items():
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predicted)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))
    cmp += 1
    
plt.title('Classifiers comparaison with ROC')
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["positive", "negative"]))


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(score)))
    plt.xticks(tick_marks, set(score), rotation=45)
    plt.yticks(tick_marks, set(score))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(y_test, prediction['Logistic'])
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cm)    

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()



