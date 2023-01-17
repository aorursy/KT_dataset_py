#Data processing packages

import pandas as pd

import numpy as np

pd.set_option('display.max_colwidth', 300)



#Visualization packages

import matplotlib.pyplot as plt

import seaborn as sns



#NLP packages

from textblob import TextBlob



import warnings

warnings.filterwarnings("ignore")
#Importing YouTube comments data

data = pd.read_csv('../input/amazon_alexa.tsv', delimiter='\t')

data
test = data.copy(deep=True)

test.loc[test['feedback'] == 1, 'feedback'] = 'Positive'

test.loc[test['feedback'] == 0, 'feedback'] = 'Negative'
fig, axs = plt.subplots(1, 2, figsize=(24, 10))



data.feedback.value_counts().plot.barh(ax=axs[0])

axs[0].set_title(("Class Distribution - Feedback {1 (positive) & 0 (negative)}"));



data.rating.value_counts().plot.barh(ax=axs[1])

axs[1].set_title("Class Distribution - Ratings");
plt.figure(figsize=(40,8))

sns.countplot("variation", hue="feedback", data=data)
from wordcloud import WordCloud



def wc(data,bgcolor,title):

    plt.figure(figsize = (50,50))

    wc = WordCloud(background_color = bgcolor, max_words = 2000, random_state=42, max_font_size = 50)

    wc.generate(' '.join(data))

    plt.imshow(wc)

    plt.axis('off')
#Displaying the first 5 rows of the data

data.head()
comm = data

comm.shape
#Calculating the Sentiment Polarity

polarity=[] # list which will contain the polarity of the comments

subjectivity=[] # list which will contain the subjectivity of the comments

for i in comm['verified_reviews'].values:

    try:

        analysis =TextBlob(i)

        polarity.append(analysis.sentiment.polarity)

        subjectivity.append(analysis.sentiment.subjectivity)

        

    except:

        polarity.append(0)

        subjectivity.append(0)
#Adding the Sentiment Polarity column to the data

comm['polarity']=polarity

comm['subjectivity']=subjectivity
comm[comm.polarity<0].head(10)
#Displaying highly positive reviews

comm[comm.polarity>0.75].head(10)
wc(comm['verified_reviews'][comm.polarity>0.75],'black','Common Words' )
#Displaying highly negative reviews

comm[comm.polarity<-0.25].head(10)
wc(comm['verified_reviews'][comm.polarity<-0.25],'black','Common Words' )
comm.polarity.hist(bins=50)
negativedata=comm[comm.polarity<-0]

negativedata.shape
negativedata["index"] = range(0,189)

negativedata = negativedata.set_index("index")

negativedata.head()
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import word_tokenize
STOPWORDS = set(stopwords.words('english'))

corpus=[]

for i in range(0,189):

    review = re.sub('[^a-zA-Z]', ' ',negativedata['verified_reviews'][i])

    review = review.lower()

    review = review.split()

    stemmer = PorterStemmer()

    review = [stemmer.stem(token) for token in review if not token in STOPWORDS]

    #contain all words that are not in stopwords dictionary

    review=' '.join(review)

    corpus.append(review)

corpus
words = []

for i in range(0,len(corpus)):

    words = words + (re.findall(r'\w+', corpus[i]))# words cantain all the words in the dataset

from collections import Counter

words_counts = Counter(words)

most_common_words = sorted(words_counts.items(), key=lambda x: x[1], reverse=True)

most_common_words
most_commmom_wordList = []

most_commmom_CountList = []

for x, y in most_common_words:

    most_commmom_wordList.append(x)

    most_commmom_CountList.append(y)
import seaborn as sns

plt.figure(figsize=(20,18))

plot = sns.barplot(np.arange(20), most_commmom_CountList[0:20])

plt.ylabel('Word Count',fontsize=20)

plt.xticks(np.arange(20), most_commmom_wordList[0:20], fontsize=20, rotation=40)

plt.title('Most Common Word used in Bad Review.', fontsize=20)

plt.show()
#Converting the polarity values from continuous to categorical

comm['polarity'][comm.polarity==0]= 0

comm['polarity'][comm.polarity > 0]= 1

comm['polarity'][comm.polarity < 0]= -1
comm.polarity.value_counts().plot.bar()

comm.polarity.value_counts()
from sklearn.feature_extraction.text import TfidfVectorizer

Vectorize = TfidfVectorizer(analyzer='word',stop_words='english',ngram_range=(1, 2),min_df=2)

X = Vectorize.fit_transform(corpus).toarray()

y = negativedata['feedback']

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, recall_score, precision_score,roc_curve,auc


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

model1 = RandomForestClassifier(n_estimators=200, max_features="auto")

model1.fit(x_train,y_train)
y_pred1 = model1.predict(x_test)

accuracy1 = accuracy_score(y_test,y_pred1)

print("Accuracy for RandomForest:\t"+str(accuracy1))

print("Precision for RandomForest:\t"+str(precision_score(y_test,y_pred1)))

print("Recall for RandomForest:\t"+str(recall_score(y_test,y_pred1)))
model2 = GradientBoostingClassifier(learning_rate=1.5, verbose=1, max_features='auto')

model2.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)

accuracy2 = accuracy_score(y_test,y_pred2)

print("Accuracy for GradientBoosting:\t"+str(accuracy2))

print("Precision for GradientBoosting:\t"+str(precision_score(y_test,y_pred2)))

print("Recall for GradientBoosting:\t"+str(recall_score(y_test,y_pred2)))