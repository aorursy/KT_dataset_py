# Importing required packages



import numpy as np # for linear algebra

import pandas as pd # for data processing, CSV file

import matplotlib.pyplot as plt # data visualization library

%matplotlib inline

import seaborn as sns # interactive visualization library built on top on matplotlib



data = pd.read_csv('/kaggle/input/train.csv') # importing training data
# Checking the head of the data

data.head()
print(len(data[data.label == 0]), 'Non-Hatred Tweets')

print(len(data[data.label == 1]), 'Hatred Tweets')

# Class distribution in this data seems to be imbalanced.

# F1 score should be used fot model performance evaluation in such situation. 
#importing different libraries for analysis, processing and classification

import nltk

from sklearn import re #regular expression for text processing

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer #word stemmer class

lemma = WordNetLemmatizer()

from wordcloud import WordCloud, STOPWORDS

from nltk import FreqDist



# Vectorizers

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression #classification model

from sklearn.metrics import confusion_matrix, classification_report, f1_score # performance evaluation criteria
def normalizer(tweet):

    tweets = " ".join(filter(lambda x: x[0]!= '@' , tweet.split()))

    tweets = re.sub('[^a-zA-Z]', ' ', tweets)

    tweets = tweets.lower()

    tweets = tweets.split()

    tweets = [word for word in tweets if not word in set(stopwords.words('english'))]

    tweets = [lemma.lemmatize(word) for word in tweets]

    tweets = " ".join(tweets)

    return tweets
data['normalized_text'] = data.tweet.apply(normalizer)
def extract_hashtag(tweet):

    tweets = " ".join(filter(lambda x: x[0]== '#', tweet.split()))

    tweets = re.sub('[^a-zA-Z]',' ',  tweets)

    tweets = tweets.lower()

    tweets = [lemma.lemmatize(word) for word in tweets]

    tweets = "".join(tweets)

    return tweets
data['hashtag'] = data.tweet.apply(extract_hashtag)
data.head()
# all tweets 

all_words = " ".join(data.normalized_text)

#print(all_words)
# Hatred tweets

hatred_words = " ".join(data[data['label']==1].normalized_text)

#print(hatred_words)
wordcloud = WordCloud(height=800, width=800, max_font_size = 110, stopwords=STOPWORDS, background_color='black')

wordcloud = wordcloud.generate(all_words)

plt.figure(figsize = (10,7))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.axis('off')

plt.show()
wordcloud = WordCloud(height=800, width=800, max_font_size = 110, stopwords=STOPWORDS, background_color='black')

wordcloud = wordcloud.generate(hatred_words)

plt.figure(figsize = (10,7))

plt.imshow(wordcloud, interpolation = "bilinear")

plt.axis('off')

plt.show()
# from nltk import FreqDist

freq_all_hashtag = FreqDist(list(" ".join(data.hashtag).split())).most_common(12)

freq_all_hashtag
freq_hatred_hashtag = FreqDist(list(" ".join(data[data['label']==1]['hashtag']).split())).most_common(12)

freq_hatred_hashtag
allhashtag = pd.DataFrame(freq_all_hashtag, columns=['words', 'frequency'])

hatredhashtag = pd.DataFrame(freq_hatred_hashtag, columns=['words', 'frequency'])

print(allhashtag.head())

print(hatredhashtag.head())
sns.barplot(x='words', y='frequency', data=allhashtag)

plt.xticks(rotation = 45)

plt.title('all hashtag words frequency')

plt.show()
sns.barplot(x='words', y='frequency', data=hatredhashtag)

plt.xticks(rotation = 45)

plt.title('hatred hashtag words frequency')

plt.show()
# to create sparse matrix corpus is created to pass to vectorizer

corpus = []

for i in range(0,len(data.id)):

    corpus.append(data['normalized_text'][i])

#corpus
cv = CountVectorizer(stop_words=stopwords.words('english'))

cv.fit(corpus)
# creating dense matrix

X = cv.transform(corpus).toarray()

y = data.iloc[:,1].values
# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
classifier1 = LogisticRegression(C=10)

classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

y_prob = classifier1.predict_proba(X_test)
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(set(data.label)))

    plt.xticks(tick_marks, set(data.label), rotation=0)

    plt.yticks(tick_marks, set(data.label))

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cm = confusion_matrix(y_test, y_pred)

print(f1_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(cm)

plot_confusion_matrix(cm)
tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=10, stop_words=stopwords.words('english'))

X1 = tfidf.fit_transform(corpus)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.33, random_state=42)

classifier2 = LogisticRegression(C=10)

classifier2.fit(X1_train, y1_train)
y1_pred = classifier2.predict(X1_test)

y1_prob = classifier2.predict_proba(X1_test)
cm1 = confusion_matrix(y1_test, y1_pred)

print(f1_score(y1_test, y1_pred))

print(classification_report(y1_test, y1_pred))

print(cm1)

plot_confusion_matrix(cm1)
threshold = np.arange(0.1,0.9,0.1)

score = [f1_score(y1_test, ((y1_prob[:,1] >= x).astype(int))) for x in threshold]
plt.plot(threshold, score)

plt.xlabel('Threshold Probability')

plt.ylabel('F1 score')

plt.show()
data2 = pd.read_csv('/kaggle/input/test.csv')

data2.head()
data2['normalized_text'] = data2['tweet'].apply(normalizer)
data2.head()

# creating corpus

corpus_test = []

for i in range(0, len(data2.id)):

    corpus_test.append(data2.normalized_text[i])

#corpus_test
Test_X = tfidf.transform(corpus_test)
pred_Y = classifier2.predict(Test_X)

prob_Y = classifier2.predict_proba(Test_X)
data2['pred_label'] = pred_Y

scores = (prob_Y[:,1] >= 0.5).astype(int)

data2['score'] = scores
data2[data2.pred_label == 1]
data2
x = True

while(x):

    tweet = input("\nTweet Something : ")

    if tweet == "exit":

        x = False

        break

    prediction = classifier2.predict(tfidf.transform([tweet]))

    if (prediction == [0]):

        print('Non-hatred Tweet')

    else:

        print('Hatred Tweet')