#import the initial libraries

import numpy as np

import pandas as pd



#plotting

import seaborn as sns

import matplotlib.pyplot as plt



#text processing

import string

import nltk

import spacy



#text visualisation

from wordcloud import WordCloud



#initializing defaults

sns.set()

%matplotlib inline
pd.set_option('display.max_columns', None)
# Train data

train_df = pd.read_csv('../input/mbti-classification/train.csv')

# Test Data

test_df = pd.read_csv('../input/mbti-classification/test.csv')
train_df.head()
test_df.head()
train_df['num posts'] = train_df['posts'].apply(lambda x:len(x.split('|||')))

test_df['num posts'] = test_df['posts'].apply(lambda x:len(x.split('|||')))

train_df.tail()
train_df['words per post'] = train_df['posts'].apply(lambda x:len(x.split()))/ train_df['num posts']

test_df['words per post'] = test_df['posts'].apply(lambda x:len(x.split()))/ test_df['num posts']

train_df.head()
train_df['num urls'] = train_df['posts'].apply(lambda x:x.count('htt'))

test_df['num urls'] = test_df['posts'].apply(lambda x:x.count('htt'))

train_df.head()
train_df['words variance post'] = train_df['posts'].apply(lambda x:np.var([len(post) for post in x.split('|||')]))/ train_df['num posts']

test_df['words variance post'] = test_df['posts'].apply(lambda x:np.var([len(post) for post in x.split('|||')]))/ test_df['num posts']

train_df.head()
figSize = (15,6)

train_df.groupby('type').sum().plot.bar(figsize=figSize)

plt.title('Summary on Personality types')

plt.ylabel('Sum')

plt.xlabel('Personality Type')
train_df.groupby('type').mean().plot.bar(figsize=figSize)

plt.legend(loc=(1,0.5))

plt.title('Summary on Personality types')

plt.ylabel('Average')

plt.xlabel('Personality Type')
train_df.groupby('type').var().plot.bar(figsize=figSize)

plt.title('Summary on Personality types')

plt.ylabel('Variance')

plt.xlabel('Personality Type')
train_df.describe()
test_df.describe()
train_df.type.value_counts().plot.bar(figsize=figSize)
train_df.type.value_counts()
#Create columns for Mind, Energy, Nature and Tactics

train_df['Mind'] = train_df.type.apply(lambda x:x[0] == 'E')

train_df['Energy'] = train_df.type.apply(lambda x:x[1] == 'N')

train_df['Nature'] = train_df.type.apply(lambda x:x[2] == 'T')

train_df['Tactics'] = train_df.type.apply(lambda x:x[3] == 'J')
#Define a list of columns to train

cols_to_train = ['Mind', 'Energy', 'Nature', 'Tactics']
#Plot the values

train_df[cols_to_train[0]].apply(lambda x: 'Extrovert' if x else 'Introvert').value_counts().plot.bar()

plt.title(cols_to_train[0])
train_df[cols_to_train[1]].apply(lambda x: 'Intuitive' if x else 'Sensing').value_counts().plot.bar()

plt.title(cols_to_train[1])
train_df[cols_to_train[2]].apply(lambda x: 'Thinking' if x else 'Feeling').value_counts().plot.bar()

plt.title(cols_to_train[2])
train_df[cols_to_train[3]].apply(lambda x: 'Judging' if x else 'Perceiving').value_counts().plot.bar()

plt.title(cols_to_train[3])
# A function to create word clouds

def visualize(label):

    '''This function creates a word cloud for each personality type to visualise the most common words'''

    words = ''

    for msg in train_df[train_df['type'] == label]['posts']:

        msg = msg.lower()

        words += msg + ' '

    wordcloud = WordCloud(width=600,height=400).generate(words)

    return wordcloud
# Displaying the Wordcloud for each personality type

types = sorted(train_df.type.unique())

fig = plt.figure(figsize=(20,30))

i = 0

for type in types:

    ax = fig.add_subplot(8,2,i+1)

    i += 1

    wordcloud = visualize(type)

    ax.imshow(wordcloud, aspect=0.8)

    ax.set_title(type, {'fontsize': 16,'fontweight' : 3})

    ax.axis('off')
def get_most_uncommon_words(type,df):

    """

        This function retrieves the most uncommon words.

        input:  type-label of the personality aspect to predict

                df- dataframe to extract the information from

        

        output: dataframe with the uncommon words from the two personality types

    """

    words = ''

    #get top 30 words for the first class

    for msg in df[df[type] == True]['posts']:

        msg = msg.lower()

        words += msg + ' '

    wordcloud = WordCloud(max_words=30).generate(words)

    #get the top 30 words for the second class

    for msg in df[df[type] == False]['posts']:

        msg = msg.lower()

        words += msg + ' '

    wordcloud1 = WordCloud(width=600,height=400,max_words=30).generate(words)

    #get the most uncommon words from the two classes

    s1 = pd.Series(dict(wordcloud.words_))

    s2 = pd.Series(dict(wordcloud1.words_))

    dtf = pd.concat([s1,s2],axis=1)

    #select only the words that appear in one personality type and return the results

    return dtf[np.sum(dtf.isnull(),axis=1) == 1]
#let us pick the second user's posts from our list

test_string = train_df.iloc[1][1]
#let us count the total number of words for the user's posts

len(test_string.split())
#let us count the total number of characters for the user's posts

len(test_string)
#let us view the actual collection of posts

test_string
def remove_links(text):

    '''This function removes links/urls and | from text'''

    #Replacing ... with a space

    text = text.replace("..."," ")

    #Replacing ||| with a space

    text = [" ".join(string.split("|||")) for string in text.split("  ") if not string.rstrip().startswith('http')]

    #Removing links

    text = [t for t in "".join(text).split() if not (t.rstrip().startswith('http') or t.rstrip().startswith("'htt"))]

    return " ".join(text)
#let us remove the links and see how many words we are left with

test_string2 = remove_links(test_string)

print(len(test_string2.split()))

print(len(test_string2))
test_string2
#Set our stemmer to PorterStemmer to reduce the words to its root stem

stemmer = nltk.stem.PorterStemmer()

#Setting our NLP model in Spacy

nlp = spacy.load('en_core_web_sm')

#Set the tokenizer

tknzr = nltk.tokenize.TweetTokenizer()

#Create the stopwords object from the Spacy NLP model

stopwords = nlp.Defaults.stop_words
def my_preprocessor(text):

   """

        This function performs text preprocessing.

        input text - string object

        output text - lower case,punctuation free, and non-stopwords containing string object 

   """

   text = remove_links(text)

   text = text.lower()

   tokens = tknzr.tokenize(text)

   tokens = [token for token in tokens if not (token in stopwords or token in string.punctuation)]

   doc = nlp(" ".join(tokens))

   tokens = [token.lemma_ for token in doc if len(token.lemma_) > 2 and not (token.lemma_[0] == '-' and token.lemma_[-1] =='-')]

   tokens = [token.replace("."," ") for token in tokens if not token[0].isnumeric()]

   text = " ".join(tokens)

   return text
#let us see what our text looks like now

test_string3 = (my_preprocessor(test_string))

len(test_string3.split())
test_string3
#Import libraries for train-test split, training the data and parameter tuning, as well as metrics to measure accuracy of predicitions

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
#Cleaning the train set

train_df.posts = train_df.posts.apply(my_preprocessor)

#Cleaning the test set

test_df.posts = test_df.posts.apply(my_preprocessor)
#Set X to the posts column

X = train_df['posts']

#Set y to the type column

y = train_df['type']
#Create train and test splits

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#Count vectorizer. We use a bigram to give more context to our words since the sentence structure has been stripped from our posts.

vect = CountVectorizer(min_df=0.1,max_df=0.7,ngram_range=(1,2))
#Create a vectorised array of the bag of words for the train and validation sets

X_train_bow = vect.fit_transform(X_train).toarray()

X_test_bow = vect.transform(X_test).toarray()

#Create the dataframes from the above arrays

X_train_bow = pd.DataFrame(X_train_bow,columns=vect.get_feature_names())

X_test_bow = pd.DataFrame(X_test_bow,columns=vect.get_feature_names())

#Creating the vectorized array from the Test data and coverting it to a dataframe

test_bow = vect.transform(test_df.posts).toarray()

test_bow = pd.DataFrame(test_bow,columns=vect.get_feature_names())
#parameters to use in a grid search for a logistic regression

log_params ={

    'C': [1.0,0.1,0.001,10],

    'class_weight': [None,'balanced'],

    'fit_intercept': [True,False],

    'penalty': ['l2','l1'],

    'tol': [0.0001,0.001,0.1]

} 
#Intialise an object for the Grid Search

gridCV = GridSearchCV(LogisticRegression(),log_params)

#Fit the data to Grid Search object

gridCV.fit(X_train_bow,y_train)
#Get the best parameters from our Grid Search

gridCV.best_params_
#use the optimal hyperparameters to instantiate the model

log_reg = LogisticRegression(C=0.001,class_weight='balanced',fit_intercept=True,tol=0.1,n_jobs=-1)
#Fit data to prediction model

log_reg.fit(X_train_bow,y_train)
#Predict the personality types using the model

preds = log_reg.predict(X_test_bow)
#Lets look at our prediction accuracy using the classification report

print(classification_report(preds,y_test))
#Quick look at the Test data

test_df.head()
#Predicting the personality types from the Test data

test_df['type'] = log_reg.predict(test_bow)
#Creating the four required columns from the 'Type' column

test_df['mind'] = test_df.type.apply(lambda x:int(x[0] == 'E'))

test_df['energy'] = test_df.type.apply(lambda x:int(x[1] == 'N'))

test_df['nature'] = test_df.type.apply(lambda x:int(x[2] == 'T'))

test_df['tactics'] = test_df.type.apply(lambda x:int(x[3] == 'J'))
#Create CSV file for submission

test_df[['id','mind','energy','nature','tactics']].to_csv('submission.csv',index=False)