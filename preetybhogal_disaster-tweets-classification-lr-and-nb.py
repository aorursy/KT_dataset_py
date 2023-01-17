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
#Import the libraries 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#preprocessing - lemmatizing, stemming

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

import re



#modelling - countvectorizing, confusion-matrix

from sklearn.metrics import confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB



#creating word cloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings

warnings.filterwarnings('ignore')
#setting the display options



pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option('display.width', None)
#read the train datset

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
#read the test dataset

kaggle_test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
#checking the train dataset

train.tail()
#checking the test dataset

kaggle_test.head()
#checking the dimension of test dataset

kaggle_test.shape
#checking the nulls in train dataset

train.isnull().sum()
#checking how balanced the two classes are

train['target'].value_counts(normalize=True)
#drop columns keyword and location 

train.drop(['keyword','location'], axis=1, inplace=True)
#checking the null

train.isnull().sum()
#checking the datatype

train.info()
#checking the text column

train.text.head()
# Printing the text lengths

lengths = [len(text) for text in train['text']]

lengths[:10]
#adding the length of the tweet to the train dataset

train['tweet_length'] = lengths

train.head()
# instantiate tokenizer

tokenizer = RegexpTokenizer(r'\w+')
#replacing certain words in the lemmatized text 

train['text']=train['text'].str.replace(r'amp','',regex=True)

train['text']=train['text'].str.replace(r'\d','',regex=True)

train['text']=train['text'].str.replace(r'Û.*.*','',regex=True)

train['text']=train['text'].str.lower()

train['text']=train['text'].str.replace('\#','',regex=True)
#tokenizing the title of working papers

train['text'] = train['text'].apply(lambda x: tokenizer.tokenize(x))
#Create stopword list

stopwords = set(STOPWORDS)

new_words = ["may","aren", "couldn", "didn", "doesn", "don", "hadn", "hasn", "haven", "isn", "let", 

                  "ll", "mustn", "re", "shan", "shouldn", "ve", "wasn", "weren", "won", "wouldn", "t",

            "within","upon", "greater","effect","new", "the","will","via","still","today","day","co",

            "one","now","year","time","yr","go","want","rt","gt","got","know","people"]

stopwords = stopwords.union(new_words)
#instantiate lemmatizer

lemmatizer = WordNetLemmatizer()
#function to lemmatize the title text

def word_lemmatizer(title):

    lem_text = " ".join([lemmatizer.lemmatize(i) for i in title if not i in stopwords])

    return lem_text
#applying the lemmatizer and checking the title column

train['text'] = train['text'].apply(lambda x: word_lemmatizer(x))

train['text'].head()
#joining all words in text column

text = " ".join(text for text in train['text'])
#wordcloud for No Disaster

no_disaster = " ".join(text for text in train[train["target"]==0]['text'])

#Create and generate a word cloud image:

wordcloud_no_disaster = WordCloud(stopwords = stopwords,collocations=False,background_color="white", max_words=150).generate(no_disaster)



# Display the generated image:

plt.imshow(wordcloud_no_disaster, interpolation='bilinear',aspect="auto")

plt.axis("off")

# store to file

plt.savefig("no_disaster_word_cloud.png", format="png")

plt.show()
#wordcloud for Disaster

disaster = " ".join(text for text in train[train["target"]==1]['text'])

#Create and generate a word cloud image:

wordcloud_disaster = WordCloud(stopwords = stopwords,collocations=False,background_color="white", max_words=150).generate(disaster)



# Display the generated image:

plt.imshow(wordcloud_disaster, interpolation='bilinear',aspect="auto")

plt.axis("off")

# store to file

plt.savefig("disaster_word_cloud.png", format="png")

plt.show()
#instantiating the count vectorizer

#fitting and transforming the title 



cv = CountVectorizer(max_df=0.8,stop_words=stopwords, max_features=10000, ngram_range=(1,3))

X = cv.fit_transform(train['text'])
#Most frequently occuring words

def get_top_n_words(text, n=None):

    vec = CountVectorizer(stop_words=stopwords).fit(train['text'])

    bag_of_words = vec.transform(train['text'])

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in      

                   vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                       reverse=True)

    return words_freq[:n]



#Convert most freq words to dataframe for plotting bar plot

top_words = get_top_n_words(train['text'], n=20)

top_df = pd.DataFrame(top_words)

top_df.columns=["Word", "Freq"]

print(top_df)



#Barplot of most freq words



sns.set(rc={'figure.figsize':(13,8)});

g = sns.barplot(x="Word", y="Freq", data=top_df);

g.set_xticklabels(g.get_xticklabels(), rotation=90);
#Most frequently occuring Bi-grams 

def get_top_n2_words(text, n=None):

    vec1 = CountVectorizer(ngram_range=(2,2),  

            max_features=2000, stop_words=stopwords).fit(train['text'])

    bag_of_words = vec1.transform(train['text'])

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in     

                  vec1.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], 

                reverse=True)

    return words_freq[:n]



top2_words = get_top_n2_words(train['text'], n=20)

top2_df = pd.DataFrame(top2_words)

top2_df.columns=["Bi-gram", "Freq"]

print(top2_df)



#Barplot of most freq Bi-grams



sns.set(rc={'figure.figsize':(13,8)});

h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df);

h.set_xticklabels(h.get_xticklabels(), rotation=90);
#defining the X and y variables 

X = train[["text"]]

y = train["target"]
#train-test split

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.25,

                                                    random_state=42,

                                                    stratify=y)
# Instantiate our CountVectorizer.

cvec = CountVectorizer(max_features=500, stop_words='english')
# Fit our CountVectorizer on the training data and transform training data.

X_train_cvec = pd.DataFrame(cvec.fit_transform(X_train['text']).toarray(),

                           columns = cvec.get_feature_names())

X_train_cvec.head()
# Transform our testing data with the already-fit CountVectorizer.

X_test_cvec = pd.DataFrame(cvec.transform(X_test['text']).toarray(),

                          columns = cvec.get_feature_names())

X_test_cvec.head()
pipe = Pipeline([

    ('cvec', CountVectorizer()),

    ('lr', LogisticRegression())

])
pipe_params = {

    'cvec__max_features': [500],

    'cvec__ngram_range': [(1,1), (1,2),(2,2)],

    'cvec__stop_words': [stopwords]

}

gs = GridSearchCV(pipe, param_grid = pipe_params, cv=5, scoring='roc_auc', verbose=1)

model_log = gs.fit(X_train['text'], y_train)

print(gs.best_score_)

# gs.best_params_
#Training score

gs.score(X_train['text'], y_train)
#Testing score

gs.score(X_test['text'], y_test)
#Predicting using X_train

y_train_preds = gs.predict(X_train['text'])

y_train_preds
#Predicting using X_test

y_test_preds = gs.predict(X_test['text'])

y_test_preds
# Generate a confusion matrix

confusion_matrix(y_test, y_test_preds)
#generating confusion matrix post logistic regression

tn, fp, fn, tp = confusion_matrix(y_test, y_test_preds).ravel()
print("True Negatives: %s" % tn)

print("False Positives: %s" % fp)

print("False Negatives: %s" % fn)

print("True Positives: %s" % tp)
# Instantiate our model

nb = MultinomialNB()
# Fit our model!

model_nb = nb.fit(X_train_cvec, y_train)
# Generate our predictions!

predictions = model_nb.predict(X_test_cvec)

predictions
# Score our model on the training set.

model_nb.score(X_train_cvec, y_train)
# Score our model on the testing set.

model_nb.score(X_test_cvec, y_test)
# Generate a confusion matrix.

confusion_matrix(y_test, predictions)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print("True Negatives: %s" % tn)

print("False Positives: %s" % fp)

print("False Negatives: %s" % fn)

print("True Positives: %s" % tp)
#checking the test dataset

kaggle_test.head()
#checking for nulls

kaggle_test.isnull().sum()
#removing the keyword and location column

kaggle_test.drop(['keyword','location'],axis=1,inplace=True)
#checking for nulls and data types

kaggle_test.info()
#checking the dimension of the dataset

kaggle_test.shape
#replacing certain words in the lemmatized text 

kaggle_test['text']=kaggle_test['text'].str.replace(r'amp','',regex=True)

kaggle_test['text']=kaggle_test['text'].str.replace(r'\d','',regex=True)

kaggle_test['text']=kaggle_test['text'].str.replace(r'Û.*.*','',regex=True)

kaggle_test['text']=kaggle_test['text'].str.lower()

kaggle_test['text']=kaggle_test['text'].str.replace('\#','',regex=True)
#tokenizing the title of working papers

kaggle_test['text'] = kaggle_test['text'].apply(lambda x: tokenizer.tokenize(x))
#applying the lemmatizer and checking the title column

kaggle_test['text'] = kaggle_test['text'].apply(lambda x: word_lemmatizer(x))

kaggle_test['text'].head()
#joining all words in text column

kaggle_text = " ".join(text for text in kaggle_test['text'])
#Create and generate a word cloud image

wordcloud_kaggle_text = WordCloud(stopwords = stopwords, collocations=False,background_color="white", max_words=150).generate(kaggle_text)



# Display the generated image

plt.imshow(wordcloud_kaggle_text, interpolation='bilinear',aspect="auto");

plt.axis("off");
#Using Logistic Regression to make predictions for kaggle_test dataset

kaggle_predictions_lr = model_log.predict(kaggle_test['text'])

kaggle_predictions_lr
#Using NB model to make predcitions for kaggle_test dataset

kaggle_cvec = cvec.transform(kaggle_test['text'])

kaggle_cvec
#converting sparse matrix to dense array

kaggle_cvec = kaggle_cvec.todense()
kaggle_predictions_nb = model_nb.predict(kaggle_cvec)
kaggle_predictions_nb
#empty dataframe

submission_lr = pd.DataFrame()
submission_lr['Id'] = kaggle_test.id

submission_lr['target'] = kaggle_predictions_lr
submission_lr.head()
# saving predictions in a csv file

submission_lr.loc[ :].to_csv('final_kaggle_lr.csv',index=False)
#empty dataframe

submission_nb = pd.DataFrame()
submission_nb['Id'] = kaggle_test.id

submission_nb['target'] = kaggle_predictions_nb
submission_nb.head()
# saving predictions in a csv file

submission_nb.loc[ :].to_csv('final_kaggle_nb.csv',index=False)