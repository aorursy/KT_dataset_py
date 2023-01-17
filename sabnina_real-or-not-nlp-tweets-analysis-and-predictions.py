import pandas as pd

import numpy as np


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
train=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

train.head(10)
test=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test.head(10)
train.info()
train.describe()
#first analysis

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

sns.countplot(x='target', data= train)

Labels= ('No Disaster', 'Real Disaster')

plt.xticks(range(2), Labels)
#analyze-tweet-sentiments-in-python

#https://www.earthdatascience.org/courses/earth-analytics-python/using-apis-natural-language-processing-twitter/analyze-tweet-sentiments-in-python/

from textblob import TextBlob

# Create textblob objects of the tweets

sentiment_objects = [TextBlob(tw) for tw in train['text']]

# Create list of polarity valuesx and tweet text

sentiment_values = [[tweet.sentiment.polarity] for tweet in sentiment_objects]



#polarity values that range from 1 to -1.

#Values closer to 1 indicate more positivity, while values closer to -1 indicate more negativity.

sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity"])

sentiment_df.head(10)
def add_polarity(_df):

    _df = pd.concat([_df, sentiment_df], axis=1)

    return _df

train=add_polarity(train)
train.head(5)
#del tweet_disaster['polarity']

#Analyze Sentiments Using Twitter Data

# Remove polarity values equal to zero

import matplotlib.pyplot as plt

sentiment_df = sentiment_df[sentiment_df.polarity != 0]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot histogram with break at zero

sentiment_df.hist(bins=[-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1],

             ax=ax,

             color="purple")



plt.title("Polarity distribution ")

plt.show()
#NLP tweets: Cleaning & Preprocessing tweets Data

import nltk

nltk.download('stopwords')

nltk.download('punkt')
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 

from nltk.stem import PorterStemmer

from textblob import Word 
stop_words = set(stopwords.words('english'))

tweet_1=[]

for tw in train["text"]:

    word_tokens = word_tokenize(tw) 

    #Delete ponctuation

    word_tokens=[word.lower() for word in word_tokens if word.isalpha()]

    #Delete stop words

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

  

    for w in word_tokens: 

        if w not in stop_words : 

            if  w!='http':

                filtered_sentence.append(w) 

               

    #print(word_tokens) 

    #print(filtered_sentence) 

    

    Stem_words = []

    ps =PorterStemmer()

    for w in filtered_sentence:

        rootWord=ps.stem(w)

        Stem_words.append(rootWord)

    #print(filtered_sentence)

    #print(Stem_words)

    #https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/

    lem=[]

    for w in filtered_sentence:

        word1 = Word(w).lemmatize("n")

        word2 = Word(word1).lemmatize("v")

        word3 = Word(word2).lemmatize("a")

        lem.append(Word(word3).lemmatize())

    tweet_1.append(lem)

     
sentiment_objects = [TextBlob(str(t)) for t in tweet_1]

sentiment_values = [[tweet_1.sentiment.polarity, str(tweet_1)] for tweet_1 in sentiment_objects]

sentiment_values[0]

sentiment_df1 = pd.DataFrame(sentiment_values, columns=["polarity_lem", "lems"])

sentiment_df1.head(10)

def add_polarity1(_df):

    _df = pd.concat([_df, sentiment_df1["lems"]], axis=1)

    return _df
train=add_polarity1(train)
train.head()
train["lems"]= train["lems"].str.replace("[", "") 

train["lems"]= train["lems"].str.replace("]", "") 

train["lems"]= train["lems"].str.replace("\'", "") 

train["lems"]= train["lems"].str.replace(",", " ") 

train.head()
# same process on test.cvs

stop_words = set(stopwords.words('english'))

tweet_2=[]

for tw in test["text"]:

    word_tokens = word_tokenize(tw) 

    #Delete ponctuation

    word_tokens=[word.lower() for word in word_tokens if word.isalpha()]

    #Delete stop words

    filtered_sentence = [w for w in word_tokens if not w in stop_words] 

    filtered_sentence = [] 

  

    for w in word_tokens: 

        if w not in stop_words : 

            if  w!='http':

                filtered_sentence.append(w) 

               

    #print(word_tokens) 

    #print(filtered_sentence) 

    

    Stem_words = []

    ps =PorterStemmer()

    for w in filtered_sentence:

        rootWord=ps.stem(w)

        Stem_words.append(rootWord)

    #print(filtered_sentence)

    #print(Stem_words)

    #https://www.analyticsvidhya.com/blog/2019/08/how-to-remove-stopwords-text-normalization-nltk-spacy-gensim-python/

    lem=[]

    for w in filtered_sentence:

        word1 = Word(w).lemmatize("n")

        word2 = Word(word1).lemmatize("v")

        word3 = Word(word2).lemmatize("a")

        lem.append(Word(word3).lemmatize())

    tweet_2.append(lem)   
sentiment_objects = [TextBlob(str(t)) for t in tweet_2]

sentiment_values = [[tweet_2.sentiment.polarity, str(tweet_2)] for tweet_2 in sentiment_objects]

sentiment_values[0]

sentiment_df1 = pd.DataFrame(sentiment_values, columns=["polarity_lem", "lems"])

sentiment_df1.head(10)
def add_polarity2(_df):

    _df = pd.concat([_df, sentiment_df1], axis=1)

    return _df

test=add_polarity2(test)
test["lems"]= test["lems"].str.replace("[", "") 

test["lems"]= test["lems"].str.replace("]", "") 

test["lems"]= test["lems"].str.replace("\'", "") 

test["lems"]= test["lems"].str.replace(",", " ") 

test.head()
#we use skcikit-learn's CountVectorizer to count the words in each tweet and turn them into data our

#machine learning model can process.

#CountVectorizer - Convert a collection of text documents to a matrix of token counts and an integer count for the number of times each word appeared in the document.

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

# Call the transform() function on one or more documents as needed to encode each as a vector.

X_train = count_vectorizer.fit_transform(train["lems"])

X_test = count_vectorizer.transform(test["lems"])
print(X_test .shape)

print(type(X_test ))

print(X_test.toarray())
#TfidfVectorizer - Convert text to word frequency vectors.

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(train["lems"])

X_test_tfidf = tfidf_vectorizer.transform(test["lems"])
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, X_train, train["target"], cv=3, scoring="f1")

scores
# Fitting Logistic Regression on Count Vectors

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logistic_reg= LogisticRegression()

scores = model_selection.cross_val_score(logistic_reg, X_train, train["target"], cv=3, scoring="f1")

scores
# Fitting Logistic Regression on tfidf

logistic_reg_tfidf = LogisticRegression()

scores = model_selection.cross_val_score(logistic_reg_tfidf, X_train_tfidf, train["target"], cv=3, scoring="f1")

scores
from sklearn.naive_bayes import MultinomialNB

multinomial_naive_bayes = MultinomialNB()

scores = model_selection.cross_val_score(multinomial_naive_bayes, X_train, train["target"], cv=3, scoring="f1")

scores
multinomial_naive_bayes.fit( X_train, train["target"])
# Fitting Naive Bayes on tfidf

multinomial_naive_bayes_tfidf = MultinomialNB()

scores = model_selection.cross_val_score(multinomial_naive_bayes_tfidf, X_train_tfidf, train["target"], cv=3, scoring="f1")

scores
# Fitting SVM on Count Vector

# Since SVM takes a lot of time, we will reduce the number of features using SVD and standardize the data before applying SVM.

# Apply SVD, 120-200 components are good enough for SVM model.



from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC



svd = TruncatedSVD(n_components=120)

xtrain_svd = svd.fit_transform(X_train)

xtest_svd= svd.transform(X_test)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl = StandardScaler()

xtrain_svd_scl = scl.fit_transform(xtrain_svd)

xtest_svd_scl = scl.transform(xtest_svd)

# Fitting a simple SVM

svm = SVC()

scores = model_selection.cross_val_score(svm , xtrain_svd_scl, train["target"], cv=3, scoring="f1")

scores
# Fitting SVM on tfidf

svd_tfidf = TruncatedSVD(n_components=120)

xtrain_svd_tfidf = svd_tfidf.fit_transform(X_train_tfidf)

xtest_svd_tfidf = svd_tfidf.transform(X_test_tfidf)



# Scale the data obtained from SVD. Renaming variable to reuse without scaling.

scl_tfidf = StandardScaler()

xtrain_svd_scl_tfidf = scl_tfidf.fit_transform(xtrain_svd_tfidf)

xtest_svd_scl_tfidf = scl_tfidf.transform(xtest_svd_tfidf)



# Fitting a simple SVM

svm_tfidf = SVC()

scores = model_selection.cross_val_score(svm_tfidf , xtrain_svd_scl_tfidf, train["target"], cv=3, scoring="f1")

scores
# Fitting xgboost on Count Vector

import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(xgb_classifier, X_train, train["target"], cv=3, scoring="f1")

scores
# Fitting xgboost on tfidf

xgb_classifier_tfidf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(xgb_classifier_tfidf , X_train_tfidf, train["target"], cv=3, scoring="f1")

scores
# Fitting xgboost on Count Vector svd feature

xgb_classifier_svd= xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(xgb_classifier_svd, xtrain_svd, train["target"], cv=3, scoring="f1")

scores
# Fitting a simple xgboost on tfidf svd features

xgb_classifier_svd_tfidf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, subsample=0.8, nthread=10, learning_rate=0.1)

scores = model_selection.cross_val_score(xgb_classifier_svd_tfidf, xtrain_svd_scl_tfidf, train["target"], cv=3, scoring="f1")

scores
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
# we choose multinomial_naive_bayes with f1=0.71

sample_submission["target"]= multinomial_naive_bayes.predict(X_test)

sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)