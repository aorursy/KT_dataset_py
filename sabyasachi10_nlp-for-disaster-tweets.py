import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.decomposition as decomposition

import seaborn as sns

import matplotlib.pyplot as plt

import regex as re

import nltk

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

from sklearn.svm import LinearSVC

from nltk.corpus import stopwords

from wordcloud import WordCloud

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import GridSearchCV,cross_val_score

from sklearn.svm import SVC
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
train_df.shape
test_df.shape
train_df.head()
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
#visualization : Disaster and non disaster tweets

%matplotlib inline

sns.countplot (x='target', data = train_df, palette = 'Blues_d')
#Convert to lower case

train_df["text"]=train_df["text"].str.lower()

test_df["text"]=test_df["text"].str.lower()

#Punctuations

symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"

for i in symbols:

    train_df["text"] = train_df["text"].str.replace(i,' ')

    test_df["text"] = test_df["text"].str.replace(i,' ')



def remove_https(text):    

    text = re.sub('https?://\S+|www\.\S+', '', text)

    return text

train_df['text'] = train_df['text'].apply(lambda x: remove_https(x))

test_df['text'] = test_df['text'].apply(lambda x: remove_https(x))



import re



emoji = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)



def strip_emoji(text):

    return emoji.sub(r'', text)



train_df['text'] = train_df['text'].apply(lambda x: strip_emoji(x))

test_df['text'] = test_df['text'].apply(lambda x: strip_emoji(x))



#Remove stop words

from nltk.corpus import stopwords

stop = stopwords.words('english')



train_df['text'] = train_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test_df['text'] = test_df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
from nltk.stem import WordNetLemmatizer



lm = WordNetLemmatizer()



def lemmatization(text):

    text = [lm.lemmatize(word,pos="v") for word in text.split()]

    return ' '.join(text)

train_df['text'] = train_df['text'].apply(lemmatization)

test_df['text'] = test_df['text'].apply(lemmatization)
#Disaster tweets dataframe

disaster_tweets =train_df[train_df.target==1]
tweets=[]

for t in disaster_tweets.text:

    tweets.append(t)

tweets[:5]
#Convert to pandas series 

disaster_tweet_text=pd.Series(tweets).str.cat(sep='')
#Remove stop words

from wordcloud import STOPWORDS

stop_words=["https","news","via","will","amp","now"] + list(STOPWORDS)
wordcloud=WordCloud(width = 1600, height = 900,max_font_size=200,stopwords=stop_words).generate(disaster_tweet_text)
plt.figure(figsize=(12,11))

plt.imshow(wordcloud)
count_vectorizer = feature_extraction.text.CountVectorizer(input = 'train_df["text"]', ngram_range=(1,2))



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text"])
print(train_vectors.shape)

print(test_vectors.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconvertor = TfidfVectorizer(max_features=5000, min_df = 5, max_df = .7,ngram_range=(1,2), stop_words= stopwords.words('english'))
train_matrix_tfidf = tfidfconvertor.fit_transform(train_df["text"])

#test_matrx_tfidf = tfidfconvertor.transform(test_df["text"])

pd.DataFrame(train_matrix_tfidf.toarray(),columns=tfidfconvertor.get_feature_names())
test_matrix_tfidf = tfidfconvertor.transform(test_df["text"])

test_matrix_tfidf.shape
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
scores = model_selection.cross_val_score(clf, train_matrix_tfidf, train_df["target"], cv=3, scoring="f1")

scores
svc_model = LinearSVC()

svc_model.fit(train_matrix_tfidf, train_df["target"])

scores = model_selection.cross_val_score(svc_model, train_matrix_tfidf, train_df["target"], cv=3, scoring="f1")

scores
params_grid=[{'kernel':['rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]},

            {'kernel':['linear'],'C':[1,10,100]}]

svm_model=GridSearchCV(SVC(),params_grid,cv=5)

svm_model1=svm_model.fit(train_matrix_tfidf,train_df["target"])
print(svm_model1.best_score_)

print(svm_model1.best_params_)
svm=SVC(kernel='rbf',gamma=0.001,C=1000)

svm.fit(train_matrix_tfidf, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = svm.predict(test_matrix_tfidf)
sample_submission.head()
sample_submission.to_csv("SVM_rbf_tfidf_ngram3.csv", index=False)
sample_submission.shape