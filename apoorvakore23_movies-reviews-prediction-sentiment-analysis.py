import string

from collections import Counter

import numpy as np

import pandas as pd

import sys

import os

import warnings

warnings.simplefilter(action = 'ignore' ,category = FutureWarning)

if not sys.warnoptions:

    warnings.simplefilter("ignore")

    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from joblib import Parallel, delayed

import string 

import time 

movies_data = pd.read_csv(r"../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip",header = 0, delimiter = '\t',encoding='utf-8')
movies_data.head()
movies_test = pd.read_csv(r"../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip",header = 0, delimiter = '\t')
movies_test.head()
len(movies_data['Phrase'].unique())
movies_data.info()
movies_test.info()
movies_data.describe()
movies_test.describe()
movies_data['Sentiment'].unique()
movies_data.isnull().sum()
movies_test.isnull().sum()
plt1 = movies_data.groupby('Sentiment')['Phrase'].count()

plt1
sns.countplot(data = movies_data , x = 'Sentiment')

plt.show()


lemma = WordNetLemmatizer() 

stopwords = stopwords.words('english')

stopwords.extend(['cinema', 'film', 'series', 'movie', 'one', 'like', 'story', 'plot'])

def clean_review(review):

    tokens = review.lower().split()

    filtered_tokens = [lemma.lemmatize(w) for w in tokens if w not in stopwords]

    return " ".join(filtered_tokens)

start_time = time.time()

clean_train_data = movies_data.copy()

clean_train_data['Phrase'] = Parallel(n_jobs=4)(delayed(clean_review)(review) for review in movies_data['Phrase'])

print(clean_train_data['Phrase'] )

end_time = time.time()

print("Cleaning Training Data Time - Processing Time = ", end_time - start_time)



# remove missing values

print("Cleaned entries: ", clean_train_data.shape[0], " out of ", movies_data.shape[0])
from sklearn.model_selection import train_test_split

target = clean_train_data.Sentiment

train_X_, validation_X_, train_y, validation_y = train_test_split(clean_train_data['Phrase'], target, test_size=0.2, random_state=22)
from sklearn.feature_extraction.text import TfidfVectorizer as tfidf



tfidf_vec = tfidf(min_df=3,  max_features=None, ngram_range=(1, 2), use_idf=1)

train_X = tfidf_vec.fit_transform(train_X_)



print("Succesfully vectorized the data.")
from sklearn.feature_extraction.text import TfidfTransformer



from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
tfidf_trans = TfidfTransformer()
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf = True ,min_df = 5 ,norm = 'max',ngram_range = (1,2),stop_words = 'english')
mx_movies = count_vectorizer.fit_transform(train_X_)

x_train_tf = tfidf_trans.fit_transform(mx_movies)

print(x_train_tf)
from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(x_train_tf,train_y)
x_test_tf =  count_vectorizer.transform(validation_X_)
y_pred1 = log.predict(x_test_tf)
y_pred1
from sklearn.metrics import classification_report
print(classification_report(validation_y,y_pred1))
from sklearn.model_selection import cross_val_score

score = cross_val_score(log , x_test_tf ,y_pred1 , scoring = 'accuracy' , cv =7)

print(" mean accuracy of the model is " , np.mean(score)*100 , np.std(score)*100)
from sklearn.svm import LinearSVC
svm = LinearSVC()
svm.fit(x_train_tf,train_y)
y_pred2 = svm.predict(x_test_tf)
y_pred2
from sklearn.metrics import classification_report
print(classification_report(validation_y,y_pred2))
from sklearn.model_selection import cross_val_score

score = cross_val_score(svm , x_test_tf ,y_pred2 , scoring = 'accuracy' , cv =7)

print(" mean accuracy of the model is " , np.mean(score)*100 , np.std(score)*100)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(max_depth = 150 , random_state = 50)
RF.fit(x_train_tf,train_y)
y_pred3 = RF.predict(x_test_tf)
y_pred3
from sklearn.metrics import classification_report
print(classification_report(validation_y,y_pred3))
from sklearn.model_selection import cross_val_score

score = cross_val_score(RF , x_test_tf ,y_pred3 , scoring = 'accuracy' , cv =7)

print(" mean accuracy of the model is " , np.mean(score)*100 , np.std(score)*100)
from sklearn.metrics import roc_curve

fpr,tpr,threshold=roc_curve(validation_y,y_pred3,pos_label=2)

plt.plot(fpr,tpr)

plt.xlabel("false Positive Rate")

plt.ylabel("True Positive Rate")

plt.plot([0,1],[0,1],'k--')

plt.show()
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train_tf ,train_y)
y_pred4 = KNN.predict(x_test_tf)
y_pred4

from sklearn.metrics import classification_report
print(classification_report(validation_y,y_pred4))
from sklearn.model_selection import cross_val_score

score = cross_val_score(KNN , x_test_tf , y_pred4, scoring = 'accuracy' , cv =7)

print(" mean accuracy of the model is " , np.mean(score)*100 , np.std(score)*100)
num_test_phrase = movies_test['Phrase'].size

clean_test_phrase = []

for i in range(0, num_test_phrase):

    if( (i+1)%10000 == 0 ):

        print ("Review %d of %d\n" % ( i+1, num_test_phrase ))

    clean_test_phrase.append(clean_review(movies_test['Phrase'][i]))
#Get a bag of words for the test set, and convert to a numpy array

test_data_features = count_vectorizer.transform(clean_test_phrase)

test_data_features = test_data_features.toarray()



#Use the random forest to make sentiment label predictions

result = RF.predict(test_data_features)



# Copy the results to a pandas dataframe with an "id" column and

# a "sentiment" column

output = pd.DataFrame( data={"PhraseId":movies_test["PhraseId"], "Sentiment":result} )



# Use pandas to write the comma-separated output file

output.to_csv( "submission.csv", index=False, quoting=3 )