import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import sqlite3

import nltk

import string

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline

import numpy as np
data = sqlite3.connect('../input/database.sqlite')

messages = pd.read_sql_query("""

SELECT Score, Id,userId

FROM Reviews

WHERE Score 

""", data)

print(messages.head(20))

messages.groupby('Score')['Id'].count().plot(kind='bar',color=['r','g','y','b'],title='Label Distribution',figsize=(10,6))

messages = pd.read_sql_query("""

SELECT 

  Score, 

  Summary,

  Userid,

  HelpfulnessNumerator as Helpnum, 

  HelpfulnessDenominator as Helpdenom

FROM Reviews  

""", data)



messages["helpFactor"] = ( messages["Helpnum"]/messages["Helpdenom"]).apply(lambda n: "useful" if n > 0.8 else "useless")

messages.groupby('helpFactor')['UserId'].count().plot(kind='pie',legend = True ,title='Label Distribution',figsize=(10,6))
#3

messages = pd.read_sql_query("""

SELECT 

  Score, 

  Summary,

  Userid,

  HelpfulnessNumerator as Helpnum, 

  HelpfulnessDenominator as Helpdenom

FROM Reviews  

""", data)

messages["helpFactor"] = ( messages["Helpnum"]/messages["Helpdenom"]).apply(lambda n: "useful" if n > 0.8 else "useless")

#messages.groupby('Score')['helpFactor'].count().plot(kind='bar',legend = True ,title='Label Distribution',figsize=(10,6))

sns.distplot(messages["Score"], kde=False, color="b")

plt.show()

messages["helpFactorNum"] = ( messages["Helpnum"]/messages["Helpdenom"])
sns.distplot(messages["Score"], hist=False, color="g", kde_kws={"shade": True})
sns.set(style="ticks")

sns.boxplot(x=messages["Score"],hue=messages["helpFactor"], data=messages, palette="Set3")

messages = pd.read_sql_query("""

SELECT 

  Score, 

  Summary,

  Text

FROM Reviews  

""", data

                      )
# Data pre-processing

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import re

import string

import nltk

cleanup_re = re.compile('<.*?>')   

cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    sentence = sentence.lower()

    sentence = cleanup_re.sub(' ', sentence).strip()

    return sentence



messages["Summary_Clean"] = messages["Summary"].apply(cleanup)

messages["Text"] = messages["Text"].apply(cleanup)

messages["Sentiment"] = messages["Score"].apply(lambda score: "positive" if score > 3 else "negative")



train, test = train_test_split(messages, test_size=0.2)

print("%d items in training data, %d in test data" % (len(train), len(test)))
from wordcloud import WordCloud, STOPWORDS



stop_words = STOPWORDS

count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 4))

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])



tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



X_new_counts = count_vect.transform(test["Summary_Clean"])

X_test_tfidf = tfidf_transformer.transform(X_new_counts)



y_train = train["Sentiment"]

y_test = test["Sentiment"]

prediction = dict()
import re

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



ps = PorterStemmer()

def preProcessString(text):

    # remove all html tags

    text = re.sub('<.*?>', ' ', str(text))

    

    # remove all special characters

    text = re.sub('[^A-Za-z0-9]+', ' ', text)

    

    # converting all text into small letters and store them as words for furthur processing

    text_list = text.lower().split()

    

    # removing stopwords from the text

    english_stop_words = set(stopwords.words('english'))

    # we have used set instead of list because, set uses hashing to store the words. So lookup is O(1).

    # where as for list the look up time is O(n) (ie., make things faster in list comprehension below)

    text_list = [word for word in text_list if word not in english_stop_words]

    

    # stemming the words (removing prefix and postfix) using Porter stemming algorithm..

    text_list = [ps.stem(word) for word in text_list]

    

    return ' '.join(text_list)
#messages["Text"] = messages["Text"].apply(preProcessString)

print(messages["Text"])
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)

mpl.rcParams['font.size']=12                

mpl.rcParams['savefig.dpi']=100         

mpl.rcParams['figure.subplot.bottom']=.1 





def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(8, 8))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()
show_wordcloud(messages[messages.Score == 1]["Summary_Clean"], title = "Negative reviews")

show_wordcloud(messages[messages.Score == 5]["Summary_Clean"], title = "Positive reviews")
train, test = train_test_split(messages, test_size=0.2)

print("%d items in training data, %d in test data" % (len(train), len(test)))
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB().fit(X_train_tfidf, y_train)

prediction['Bernoulli'] = model.predict(X_test_tfidf)
# Print the values precision , recall and F1-score for Naive Bayes classification

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc,roc_auc_score

print("The classification metric values for Bernoulli naive bayes classification are ")

print(metrics.classification_report(y_test, prediction['Bernoulli'], target_names = ["positive", "negative"]))

print("The connfusion matrix for Bernoulli NB ")

print(confusion_matrix(y_test,prediction['Bernoulli']))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e6)

logreg_result = logreg.fit(X_train_tfidf, y_train)

prediction['Logistic'] = logreg.predict(X_test_tfidf)
# Print the values precision , recall and F1-score for Logistic regression

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

print("The classification metric values for Logistic regression are ")

print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["positive", "negative"]))
print("The connfusion matrix for Bernoulli NB ")

print(confusion_matrix(y_test,prediction['Logistic']))

from sklearn.svm import LinearSVC

resultSVC = LinearSVC( max_iter=1000)

resultSVCClassifier = resultSVC.fit(X_train_tfidf , y_train)

prediction['LinearSVC'] = resultSVC.predict(X_test_tfidf)
# Print the values precision , recall and F1-score for Linear SVC

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

print("The classification metric values for linear SVM classification are ")

print(metrics.classification_report(y_test, prediction['LinearSVC'], target_names = ["positive", "negative"]))
print("The connfusion matrix for Bernoulli NB ")

print(confusion_matrix(y_test,prediction['']))

from sklearn.svm import SVC

resultSVC = SVC(kernel='rbf')

resultSVCClassifier = resultSVC.fit(X_train_tfidf , y_train)

prediction['rbfSVC'] = resultSVC.predict(X_test_tfidf )
# Print the values precision , recall and F1-score for RBF SVC

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

print("The classification metric values for RBF SVM classification are ")

print(metrics.classification_report(y_test, prediction['rbfSVC'], target_names = ["positive", "negative"]))