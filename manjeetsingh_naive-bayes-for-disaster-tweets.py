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
import os

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import re

import nltk

from matplotlib import pyplot as plt

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
import pandas as pd

test = pd.read_csv("../input/nlp-getting-started/test.csv")

train = pd.read_csv("../input/nlp-getting-started/train.csv")
import re



test_str = train.loc[417, 'text']



def clean_text(text):

    text = re.sub(r'https?://\S+', '', text) # Remove link

    text = re.sub(r'\n',' ', text) # Remove line breaks

    text = re.sub('\s+', ' ', text).strip() # Remove leading, trailing, and extra spaces

    return text



print("Original text: " + test_str)

print("Cleaned text: " + clean_text(test_str))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



remove_emoji("Omg another Earthquake 😔😔")

train['text']=train['text'].apply(lambda x: remove_emoji(x))

test['text']=test['text'].apply(lambda x: remove_emoji(x))
# Stemming and Lemmatization examples

text = "feet cats wolves talked"



tokenizer = nltk.tokenize.TreebankWordTokenizer()

tokens = tokenizer.tokenize(text)



# Stemmer

stemmer = nltk.stem.PorterStemmer()

print("Stemming the sentence: ", " ".join(stemmer.stem(token) for token in tokens))



# Lemmatizer

lemmatizer=nltk.stem.WordNetLemmatizer()

print("Lemmatizing the sentence: ", " ".join(lemmatizer.lemmatize(token) for token in tokens))
import string

def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



example="#flood #disaster Heavy rain causes flash flooding of streets in Manitou, Colorado Springs areas"

train['text']=train['text'].apply(lambda x : remove_punct(x))

test['text']=test['text'].apply(lambda x : remove_punct(x))
def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)

print(remove_html(example))

train['text']=train['text'].apply(lambda x : remove_html(x))

test['text']=test['text'].apply(lambda x : remove_html(x))
def find_hashtags(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"#\w+", tweet)]) or 'no'



def find_mentions(tweet):

    return " ".join([match.group(0)[1:] for match in re.finditer(r"@\w+", tweet)]) or 'no'



def find_links(tweet):

    return " ".join([match.group(0)[:] for match in re.finditer(r"https?://\S+", tweet)]) or 'no'



def process_text(df):

    

    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))

    df['hashtags'] = df['text'].apply(lambda x: find_hashtags(x))

    df['mentions'] = df['text'].apply(lambda x: find_mentions(x))

    df['links'] = df['text'].apply(lambda x: find_links(x))

    # df['hashtags'].fillna(value='no', inplace=True)

    # df['mentions'].fillna(value='no', inplace=True)

    

    return df

    

train = process_text(train)

test = process_text(test)
from wordcloud import STOPWORDS



def create_stat(df):

    # Tweet length

    df['text_len'] = df['text'].apply(len)

    # Word count

    df['word_count'] = df["text"].apply(lambda x: len(str(x).split()))

    # Stopword count

    df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

    # Punctuation count

    df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

    # Count of hashtags (#)

    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(str(x).split()))

    # Count of mentions (@)

    df['mention_count'] = df['mentions'].apply(lambda x: len(str(x).split()))

    # Count of links

    df['link_count'] = df['links'].apply(lambda x: len(str(x).split()))

    # Count of uppercase letters

    df['caps_count'] = df['text_clean'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))

    # Ratio of uppercase letters

    df['caps_ratio'] = df['caps_count'] / df['text_len']

    return df



train = create_stat(train)

test = create_stat(test)

print(train.shape, test.shape)
cols = ['text_clean','hashtags','mentions','links','text_len','word_count','stop_word_count','punctuation_count','hashtag_count','mention_count','link_count','caps_count','caps_ratio']

train.drop(cols, axis=1, inplace=True)

test.drop(cols, axis=1, inplace=True)
corpus  = []

pstem = PorterStemmer()

for i in range(train['text'].shape[0]):

    #Remove unwanted words

    text = re.sub("[^a-zA-Z]", ' ', train['text'][i])

    #Transform words to lowercase

    text = text.lower()

    text = text.split()

    #Remove stopwords then Stemming it

    text = [pstem.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    #Append cleaned tweet to corpus

    corpus.append(text)
#Create dictionary 

uniqueWords = {}

for text in corpus:

    for word in text.split():

        if(word in uniqueWords.keys()):

            uniqueWords[word] += 1

        else:

            uniqueWords[word] = 1

            

#Convert dictionary to dataFrame

uniqueWords = pd.DataFrame.from_dict(uniqueWords,orient='index',columns=['WordFrequency'])

uniqueWords.sort_values(by=['WordFrequency'], inplace=True, ascending=False)

print("Number of records in Unique Words Data frame are {}".format(len(uniqueWords)))

uniqueWords.head(10)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = len(uniqueWords))

#Create Bag of Words Model , here X represent bag of words

X = cv.fit_transform(corpus).todense()

y = train['target'].values
#Split the train data set to train and test data

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2, random_state=2020)
# Fitting multinomial naive bayes Model to the Training set

classifiermnb = MultinomialNB()

classifiermnb.fit(X_train, y_train)

y_predmnb = classifiermnb.predict(X_test)

confusionmatrix_mnb = confusion_matrix(y_test, y_predmnb)

confusionmatrix_mnb
print('MultinomialNB Model Accuracy Score for Train Data set is {}'.format(classifiermnb.score(X_train, y_train)))

print('MultinomialNB Model Accuracy Score for Test Data set is {}'.format(classifiermnb.score(X_test, y_test)))

print('MultinomialNB Model F1 Score is {}'.format(f1_score(y_test, y_predmnb)))
# Fitting Logistic Regression Model to the Training set

classifier_lr = LogisticRegression()

classifier_lr.fit(X_train, y_train)

y_predlr = classifier_lr.predict(X_test)

confusionmatrix_lr = confusion_matrix(y_test, y_predlr)

confusionmatrix_lr
#Model Accuracy

print('Logistic Regression Model Accuracy Score for Train Data set is {}'.format(classifier_lr.score(X_train, y_train)))

print('Logistic Regression Model Accuracy Score for Test Data set is {}'.format(classifier_lr.score(X_test, y_test)))

print('Logistic Regression Model F1 Score is {}'.format(f1_score(y_test, y_predlr)))
print("Number of records present in Test Data Set are {}".format(len(test.index)))

print("Number of records without keywords in Test Data are {}".format(len(test[pd.isnull(test['keyword'])])))

print("Number of records without location in Test Data are {}".format(len(test[pd.isnull(test['location'])])))
#Fitting into test set

X_testset=cv.transform(test['text']).todense()

y_test_pred_mnb = classifiermnb.predict(X_testset)
#Fetching Id to differnt frame

y_test_id=test[['id']]

y_test_id=y_test_id.values

y_test_id=y_test_id.ravel()
y_test_pred_mnb=y_test_pred_mnb.ravel()

submission_df_mnb=pd.DataFrame({"id":y_test_id,"target":y_test_pred_mnb})

submission_df_mnb.set_index("id")
submission_df_mnb.to_csv("submission_mnb.csv",index=False)
submission_file_path = "../input/nlp-getting-started/submission_mnb.csv"