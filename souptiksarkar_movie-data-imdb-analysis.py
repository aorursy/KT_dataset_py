import numpy as np
import scipy as sc
import pandas as np
import matplotlib.pyplot as plt
import pandas as pd
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import seaborn as sns
df=pd.read_csv('../input/movie_reviews.csv')
number=LabelEncoder()
df['sentiment']=number.fit_transform(df['sentiment'])
reviews=df['review']
sentiments=df['sentiment']
reviews=np.array(reviews)
sentiments=np.array(sentiments)
df.head()

tokenizer=ToktokTokenizer()
stopwords_list=nltk.corpus.stopwords.words('english')
stopwords_list.remove('no')
stopwords_list.remove('not')
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text
df['review']=df['review'].apply(strip_html_tags)
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text
df['review']=df['review'].apply(remove_accented_chars)
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)
    return text
df['review']=df['review'].apply(remove_special_characters)
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
df['review']=df['review'].apply(simple_stemmer)
stop=set(stopwords.words('english'))
print(stop)

#removing the stopwords
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
#Apply function on review column
df['review']=df['review'].apply(remove_stopwords)
#division into train and test set

norm_train_reviews=df.review[:40000]
norm_train_reviews
norm_test_reviews=df.review[40000:]
norm_test_reviews
####FEATURE ENGINEERING
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
tv_train=tf.fit_transform(norm_train_reviews)
tv_train
tv_test=tf.transform(norm_test_reviews)
train_sentiments=df.sentiment[:40000]
test_sentiments=df.sentiment[40000:]
train_sentiments
#fitting of logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',max_iter=500)
lr.fit(tv_train,train_sentiments)
lr.score(tv_train,train_sentiments)
predict=lr.predict(tv_test)
lr_tfidf_score=accuracy_score(test_sentiments,predict)
print("lr_tfidf_score :",lr_tfidf_score)
cm_tfidf=confusion_matrix(test_sentiments,predict,labels=[1,0])
print(cm_tfidf)
plt.figure(figsize=(9,9))
sns.heatmap(cm_tfidf, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(lr_tfidf_score)
plt.title(all_sample_title, size = 15);
####Use of Unsupervised method of SA in nlp(AFINN LEXICONS)
from afinn import Afinn
af=Afinn()
sentiment_scores=[af.score(article) for article in df['review']]
sentiment_category=['positive' if score>0
                   else 'negative' if score<0
                   else 'neutral' for score in sentiment_scores]
df2=pd.DataFrame([list(df['sentiment']),sentiment_scores,sentiment_category]).T
df2.head()
import random
random.seed(10)
df2.columns=['sentiment','sentiment_scores','sentiment_category']
df2['sentiment_scores']=df2.sentiment_scores.astype('float')
df2.head()
df2.groupby(by=['sentiment']).describe()
###visualisation of AFINN lexicons

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sp = sns.stripplot(x='sentiment', y="sentiment_scores", 
                   hue='sentiment', data=df2, ax=ax1)
bp = sns.boxplot(x='sentiment', y="sentiment_scores", 
                 hue='sentiment', data=df2, palette="Set2", ax=ax2)
t = f.suptitle('Visualizing News Sentiment', fontsize=14)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
scores=sentiment_analyzer_scores(df['review'])


