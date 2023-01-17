#Importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#reading the data
data=pd.read_csv('amazon_alexa.tsv', sep='\t')
data.info()
#Checking for missing data
data.isnull().any().any()
#Determine overall sentiment using histogram to plot feedback
overall_sentiment = data['feedback']
plt.hist(overall_sentiment, bins = 2)
plt.xlabel('Negative             Positive ')
plt.ylabel('Number of Reviews')
plt.show() 
data.feedback.value_counts()
data.groupby('variation').mean()[['rating']].plot.barh(figsize=(12, 7),colormap = 'ocean')
plt.title("Variation wise Mean Ratings");
Sentiment_count=data.groupby('rating').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['verified_reviews'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()
data.rating.value_counts()
# adding a length column for analyzing the length of the reviews

data['length'] = data['verified_reviews'].apply(len)

data.groupby('length').describe().sample(5)
color = plt.cm.rainbow(np.linspace(0, 1, 15))
data['variation'].value_counts().plot.bar(color = color, figsize = (15, 9))
plt.title('Distribution of Variations in Alexa', fontsize = 20)
plt.xlabel('variations')
plt.ylabel('count')
plt.show()
#finding which words occur the most
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(stop_words='english',tokenizer = token.tokenize)
words = cv.fit_transform(data.verified_reviews)
sum_words = words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
color = plt.cm.jet(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Top 20 Most Frequently Occuring Words")
plt.show()
#todo:wordcloud
!pip install wordcloud
from wordcloud import WordCloud
wordcloud = WordCloud(background_color='white',width=800, height=500).generate_from_frequencies(dict(words_freq))
plt.figure(figsize=(10,8))
plt.imshow(wordcloud)
plt.title("WordCloud - Vocabulary from Reviews", fontsize=22);
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', data['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['verified_reviews'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['rating'], test_size=0.3, random_state=123)
#model building, model used: Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted)*100)
text_counts= cv.fit_transform(data['verified_reviews'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_counts, data['rating'], test_size=0.3, random_state=1)
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted)*100)
predicted