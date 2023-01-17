# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')

data.head()
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

data.head()
data.shape #Shape of the data
data.v1.value_counts() #Total number of ham and spam messages
data.v1.value_counts().plot(kind = 'bar', grid = 'true')

plt.show() #Bar chart showing the total number of ham and spam messages
#Balancing the data set

ham   = data.loc[data.v1 == 'ham']

spam  = data.loc[data.v1 == 'spam']

spam2 = data.loc[data.v1 == 'spam']

spam3 = data.loc[data.v1 == 'spam']

spam4 = data.loc[data.v1 == 'spam']

spam5 = data.loc[data.v1 == 'spam']

spam6 = data.loc[data.v1 == 'spam']

spam7 = data.loc[data.v1 == 'spam'][:343]

df = pd.concat([ham, spam, spam2, spam3, spam4, spam5, spam6, spam7]) #Concat the data set

df = df.sample(frac=1).reset_index(drop = True)

df.v1.value_counts()
df.isna().sum() #Checking for missing values
#Text Cleaning

from spacy.lang.en.stop_words import STOP_WORDS as stopwords

print(stopwords)
#Length of stopwords

df['stop_words_len'] = df['v2'].apply(lambda x: len([t for t in x.split() if t in stopwords]))

df.head(2)
df['hashtags'] = df['v2'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))

df['mention_cs'] = df['v2'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))

df.head(4)
#Checking for hashtags and mentions

df.hashtags.value_counts()
df.mention_cs.value_counts()
#Lower case conversion

df['v2'] = df['v2'].apply(lambda x:str(x).lower())

df.sample(3)
#contraction to expansion

contractions = {

  "ain't": "am not",

  "aren't": "are not",

  "can't": "cannot",

  "can't've": "cannot have",

  "'cause": "because",

  "could've": "could have",

  "couldn't": "could not",

  "couldn't've": "could not have",

  "didn't": "did not",

  "doesn't": "does not",

  "don't": "do not",

  "hadn't": "had not",

  "hadn't've": "had not have",

  "hasn't": "has not",

  "haven't": "have not",

  "he'd": "he would",

  "he'd've": "he would have",

  "he'll": "he will",

  "he'll've": "he will have",

  "he's": "he is",

  "how'd": "how did",

  "how'd'y": "how do you",

  "how'll": "how will",

  "how's": "how is",

  "I'd": "I would",

  "I'd've": "I would have",

  "I'll": "I will",

  "I'll've": "I will have",

  "I'm": "I am",

  "I've": "I have",

  "isn't": "is not",

  "it'd": "it had",

  "it'd've": "it would have",

  "it'll": "it will",

  "it'll've": "it will have",

  "it's": "it is",

  "let's": "let us",

  "ma'am": "madam",

  "mayn't": "may not",

  "might've": "might have",

  "mightn't": "might not",

  "mightn't've": "might not have",

  "must've": "must have",

  "mustn't": "must not",

  "mustn't've": "must not have",

  "needn't": "need not",

  "needn't've": "need not have",

  "o'clock": "of the clock",

  "oughtn't": "ought not",

  "oughtn't've": "ought not have",

  "shan't": "shall not",

  "sha'n't": "shall not",

  "shan't've": "shall not have",

  "she'd": "she would",

  "she'd've": "she would have",

  "she'll": "she will",

  "she'll've": "she will have",

  "she's": "she is",

  "should've": "should have",

  "shouldn't": "should not",

  "shouldn't've": "should not have",

  "so've": "so have",

  "so's": "so is",

  "that'd": "that would",

  "that'd've": "that would have",

  "that's": "that is",

  "there'd": "there had",

  "there'd've": "there would have",

  "there's": "there is",

  "they'd": "they would",

  "they'd've": "they would have",

  "they'll": "they will",

  "they'll've": "they will have",

  "they're": "they are",

  "they've": "they have",

  "to've": "to have",

  "wasn't": "was not",

  "we'd": "we had",

  "we'd've": "we would have",

  "we'll": "we will",

  "we'll've": "we will have",

  "we're": "we are",

  "we've": "we have",

  "weren't": "were not",

  "what'll": "what will",

  "what'll've": "what will have",

  "what're": "what are",

  "what's": "what is",

  "what've": "what have",

  "when's": "when is",

  "when've": "when have",

  "where'd": "where did",

  "where's": "where is",

  "where've": "where have",

  "who'll": "who will",

  "who'll've": "who will have",

  "who's": "who is",

  "who've": "who have",

  "why's": "why is",

  "why've": "why have",

  "will've": "will have",

  "won't": "will not",

  "won't've": "will not have",

  "would've": "would have",

  "wouldn't": "would not",

  "wouldn't've": "would not have",

  "y'all": "you all",

  "y'alls": "you alls",

  "y'all'd": "you all would",

  "y'all'd've": "you all would have",

  "y'all're": "you all are",

  "y'all've": "you all have",

  "you'd": "you had",

  "you'd've": "you would have",

  "you'll": "you you will",

  "you'll've": "you you will have",

  "you're": "you are",

  "you've": "you have"

}
def count_to_exp(x):

    if type(x) is str:

        for key in contractions:

            value = contractions[key]

            x = x.replace(key, value)

        return x

    else:

        return x
#Applying contraction to expantion code on dataframe(df)

df['v2'] = df['v2'].apply(lambda x: count_to_exp(x))

df.head(5)
#Punctuation and special characters removal

import re

df['v2'] = df['v2'].apply(lambda x: re.sub(r'[^\w ]+','', x ))

df.head(5)
#Removing multiple spaces

df['v2'] = df['v2'].apply(lambda x: ' '.join(x.split()))

df.head(2)
#Numeric digits count

df['numeric_count'] = df['v2'].apply(lambda x:len([t for t in x.split() if t.isdigit()]))

df.head(4)
#Tokenization

import re

def tokenize(txt):

    tokens = re.split('\W+', txt)

    return tokens

df['text_tokenized'] = df['v2'].apply(lambda x: tokenize(x))

df.head()
#Stopwords removal

df['no_stop_w'] = df['v2'].apply(lambda x: ' '.join([t for t in x.split() if t not in stopwords]))

df.head(2)
import wordcloud

from wordcloud import WordCloud

text = ' '.join(df['no_stop_w'])

len(text)
wc = WordCloud(width=1000, height = 500).generate(text)

plt.imshow(wc)

plt.axis('off')

plt.show()
X = df['no_stop_w']

y = df['v1']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
#ML libraries

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

#Creating a pipeline 

message_clf = Pipeline([

                    ('vect', TfidfVectorizer()), 

                    ('clf', SVC(probability = True)),

                    ])
message_clf.fit(x_train, y_train)
message_clf.score(x_test, y_test)
y_pred = message_clf.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))