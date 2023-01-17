# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Let's do some basics import and CountVectorizer so to call the transform() function on one or more documents as needed to encode each as a vector
from os import path
from pandas import DataFrame
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import re
# Let's import some NLP modules such as PorterStemmer, SnowballStemmer, WordNetLemmetizer
# download vader_lexicon, and stopwords

import nltk
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer    # Lemmatization is similar to stemming but it brings context to the words. So it links words with similar meaning to one word.
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
# Let's import some visualization modules

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import style
import matplotlib.colors
import wordcloud   # Sentiment-based Word Clouds
from wordcloud import WordCloud, STOPWORDS 
from PIL import Image
# Change and set directory to kaggle/input

os.chdir('/kaggle/input')
os.getcwd()
# Let's read IMDB Dataset and store it into a dataframe "df"

df=pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv',header=0,error_bad_lines=True,encoding='utf8')

df.dtypes
# Let's look at our table
df.head()
# Let's define a function "sc" to run sentimental analysis on the text "review" and return the compound value (-1 to +1)
def sc(x):
    score=SentimentIntensityAnalyzer().polarity_scores(x)
    return score['compound']
## Let's apply the compound score of our sentimental analysis to "review" storing the results in a new column "SentScore" through 
# map function

df["SentScore"]=df["review"].map(sc)
# Let's look at our updated table 
df.head()
# Let's define a function "sc" to run sentimental analysis on the text "review" and return the compound value (-1 to +1)


def sca(lb):
    if lb >= .6:
        return "Very Good"
    elif (lb > .2) and (lb < .6):
        return "Good"
    elif (lb > -.2) and (lb < .2):
        return "Average"
    elif (lb > -.6) and (lb < -.2):
        return "Disappointing"
     
    else:
        return "Regrettable"
# Now we insert a column to indicate the class of the review ("Very Good" , "Good", "Average", "Disappointing", "Regrettable")

df["SentClass"]=df["SentScore"].map(sca)
# Let's check our updated table

df.head(15)
# We define a function for which relatively to the "sentiment" column, positive=1 | negative=0

def num(lb):
    if lb == 'positive':
        return 1   
    else:
        return 0
# let's create a new column "sentiment_bin" applying the function above using .map

df["sentiment_bin"]=df["sentiment"].map(num)
# Let's check the updated table

df.head(15)
# Similarly to what we did above, for the SentScore results (-1 to +1) we define a function for which a value >= 0 equals 1(positive), else 0(negative)

def numscore(lb):
    if lb >= 0:
        return 1     
    else:
        return 0
# let's create a new column "SentScore_bin" applying the function above using .map

df["SentScore_bin"]=df["SentScore"].map(numscore)
# Let's check the updated table

df.head(15)
# Let's do now some TEXT ADJUSTMENTS / CLEANING
# Make text lower case
df["review"]  = df["review"].str.lower()
# Remove digits from text
def Remove_digit(text):
    result = re.sub(r"\d", "", text)
    return result
# Remove HTML from text
def remove_html(text):
    result = re.sub(r'<.*?>','',text) # Find out anything that is in between < & > symbol 
    return result
# Remove special text characters
def remove_spl(text):
    result = re.sub(r'\W',' ',text) 
    return result
# Link words with similar meaning to one word (in context)
def lem_word(text):
    result= WordNetLemmatizer().lemmatize(text)
    return result
# Let's apply all of the above functions to the text column "review"

df["review"]  = df["review"].apply(Remove_digit)
df["review"]  = df["review"].apply(remove_html)
df["review"]  = df["review"].apply(remove_spl)
df["review"]  = df["review"].apply(lem_word)
# Let's check the updated table

df.head()
# Let's store the adjusted text to the object 'corpus1' and transform it into a List
corpus1=df['review'].tolist()
# Let's create an object "corpus" that includes the first 1000 values of the list 'corpus1', otherwise the machine could take too long to run the command

corpus=corpus1[ :1000]
# Count Vectorisation
# I have defined ngram range to be unigrams and bigrams (it starts with one word and goes up to two when vectorizing)

from sklearn.feature_extraction import text

cv = text.CountVectorizer(input=corpus,ngram_range=(1,2),stop_words='english')
matrix = cv.fit_transform(corpus)

# I am converting the matrix_cv into a dataframe 
corpus2 = pd.DataFrame(matrix.toarray(), columns=cv.get_feature_names())
# Let's take a snapshot at the data
corpus2.head()
# One thing to notice here is the dimension of this data
# We have 1000 documents (rows) which is consistent with the selected amount of rows of our list corpus 
# and 110012 columns which is humangous. We have a created a giant matrix

# It is noticeable that many features contain 0, since not all words willbe present across documents of the corpus(2)

corpus2.shape

# TF-IDF, Term Frequency and Inverse Document Freq
# We run a TF-IDF representation on the same corpus, same like before and also this time removing the english stop_words


tf = text.TfidfVectorizer(input=corpus, ngram_range=(1,2),stop_words='english')

matrix1 = tf.fit_transform(corpus)

# I am converting the matrix1 into a dataframe X
X = pd.DataFrame(matrix1.toarray(), columns=tf.get_feature_names())
# Let's take a look at our matrix X

X.head()
# Let's set our y to be the first 1000 values of the column SentScore_bin (based on our sentiment analysis)

y = df['SentScore_bin'][:1000].values
# Let's take a look at the array 'y'
print(y)
# We are going to try and run the RandomForest Classifier on X= vectorized matrix and y= SentScore_bin
# using the fit transformation of the tf - idf matrix to array =X

# Let's split X and y in training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)
# Let's set the RandomForestClassifier and set the parameters
# Let's fit the model on X and y training data

from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=4, min_samples_split=9, n_estimators=100)
text_classifier.fit(X_train, y_train)
# Let's run the prediction on the X test data and store them into the object 'predictions'

predictions = text_classifier.predict(X_test)
# We can see that running the RANDOM FOREST CLASSIFIER we get an accuracy score of 68%. NOT amazing!

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
# Let's try LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
#training the model
lr=LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,l1_ratio=None,max_iter=100,
multi_class='auto',n_jobs=None,penalty='l2',random_state=23,solver='lbfgs',tol=0.0001,verbose=0,warm_start=False)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(X_train,y_train)
print(lr_tfidf)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(X_test)
print(lr_tfidf_predict)
# Accuracy score running a LOGISTIC REGRESSION is pretty low.....only 61.5%!

#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
#Classification report for tfidf features
lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])
print(lr_tfidf_report)
# GRADIENT BOOSTING CLASSIFIER
from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(n_estimators=80,random_state=23)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators': [80,100,120,140,160]})
mod.fit(X_train,y_train)
mod.best_estimator_
clf=GradientBoostingClassifier(n_estimators=100,random_state=23)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
clf.feature_importances_
feature_imp=pd.Series(clf.feature_importances_)
feature_imp.sort_values(ascending=False)
# Let's now repeat the operations setting the first 1000 values of our column 'sentiment_bin' as our "y"
y = df['sentiment_bin'][:1000].values
# We split again in training and test data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=23)
# RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
text_classifier=RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.3, min_samples_leaf=4, min_samples_split=9, n_estimators=100)
text_classifier.fit(X_train, y_train)
predictions = text_classifier.predict(X_test)
# It seems to be more accurate with a score of 78%!

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  
print(accuracy_score(y_test, predictions))
# Let's try again LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
#training the model
lr=LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,l1_ratio=None,max_iter=100,
multi_class='auto',n_jobs=None,penalty='l2',random_state=23,solver='lbfgs',tol=0.0001,verbose=0,warm_start=False)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(X_train,y_train)
print(lr_tfidf)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(X_test)
print(lr_tfidf_predict)
# Logistic regression with y=sentiment_bin gives us the highest accuracy----81%! Not bad

#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
#Classification report for tfidf features
lr_tfidf_report=classification_report(y_test,lr_tfidf_predict,target_names=['0','1'])
print(lr_tfidf_report)

# Let's try with the GRADIENT BOOSTING CLASSIFIER

from sklearn.ensemble import GradientBoostingClassifier
clf=GradientBoostingClassifier(n_estimators=80,random_state=23)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV
mod=GridSearchCV(clf,param_grid={'n_estimators': [80,100]})
mod.fit(X_train,y_train)
mod.best_estimator_
clf=GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.1, loss='deviance', max_depth=3,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_iter_no_change=None, presort='deprecated',
                           random_state=23, subsample=1.0, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)
clf.fit(X_train,y_train)
clf.score(X_test,y_test)