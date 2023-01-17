# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from nltk.corpus import stopwords



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_columns', None) 

pd.set_option('display.max_rows', None)  

pd.set_option('display.max_colwidth', -1) 
# reading the csv file into pandas dataframes

df=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()
df['target'].value_counts()
#creating a new column- length 

# this gives the length of the post

df['length'] = np.NaN

for i in range(0,len(df['text'])):

    df['length'][i]=(len(df['text'][i]))

df.length = df.length.astype(int)
df.head()
#creating subplots to see distribution of length of tweet

sns.set_style("darkgrid");

f, (ax1, ax2) = plt.subplots(figsize=(12,6),nrows=1, ncols=2,tight_layout=True);

sns.distplot(df[df['target']==1]["length"],bins=30,ax=ax1);

sns.distplot(df[df['target']==0]["length"],bins=30,ax=ax2);

ax1.set_title('\n Distribution of length of tweet labelled Disaster \n');

ax2.set_title('\n Distribution of length of tweet labelled No Disaster \n');

ax1.set_ylabel('Frequency');
# word cloud for words related to Disaster 

text=" ".join(post for post in df[df['target']==1].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Frequntly occuring words related to Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
# word cloud for words related to No Disaster 

text=" ".join(post for post in df[df['target']==0].text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('Frequntly occuring words related to No Disaster \n\n',fontsize=18)

plt.axis("off")

plt.show()
#Calculating basline accuracy

df['target'].value_counts(normalize=True)
# Import Tokenizer

from nltk.tokenize import RegexpTokenizer

# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+') 

# Changing the contents of selftext to lowercase

df.loc[:,'text'] = df.text.apply(lambda x : str.lower(x))

# Removing hyper link, latin characters and digits

df['text']=df['text'].str.replace('http.*.*', '',regex = True)

df['text']=df['text'].str.replace('û.*.*', '',regex = True)

df['text']=df['text'].str.replace(r'\d+','',regex= True)

# "Run" Tokenizer

df['tokens'] = df['text'].map(tokenizer.tokenize)
#displaying first 5 rows of dataframe

df.head()
# Printing English stopwords

print(stopwords.words("english"))
#assigning stopwords to a variable

stop = stopwords.words("english")
# adding this stop word to list of stopwords as it appears on frequently occuring word

item=['amp'] #'https','co','http','û','ûò','ûó','û_'
stop.extend(item)
#removing stopwords from tokens

df['tokens']=df['tokens'].apply(lambda x: [item for item in x if item not in stop])
# When we "lemmatize" data, we take words and attempt to return their lemma, or the base/dictionary form of a word.

# Importing lemmatizer 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

# Instantiating lemmatizer 

lemmatizer = WordNetLemmatizer()



lemmatize_words=[]

for i in range (len(df['tokens'])):

    word=''

    for j in range(len(df['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(df['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list



#creating a new column to store the result

df['lemmatized']=lemmatize_words

#displaying first 5 rows of dataframe

df.head()
#reading the test data

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
test.head()
#creating a new column- length 

# this gives the length of the post

test['length'] = np.NaN

for i in range(0,len(test['text'])):

    test['length'][i]=(len(test['text'][i]))

test.length = test.length.astype(int)
# word cloud for Frequntly occuring words in test dataframe

text=" ".join(post for post in df.text)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\nFrequntly occuring words in test dataframe \n\n',fontsize=18)

plt.axis("off")

plt.show()
# Import Tokenizer

from nltk.tokenize import RegexpTokenizer

# Instantiate Tokenizer

tokenizer = RegexpTokenizer(r'\w+') 

# Changing the contents of selftext to lowercase

test.loc[:,'text'] = test.text.apply(lambda x : str.lower(x))

# Removing hyper link, latin characters and digits

test['text']=test['text'].str.replace('http.*.*', '',regex = True)

test['text']=test['text'].str.replace('û.*.*', '',regex = True)

test['text']=test['text'].str.replace(r'\d+','',regex= True)

# "Run" Tokenizer

test['tokens'] = test['text'].map(tokenizer.tokenize)
#displaying first 5 rows of dataframe

test.head()
#removing stopwords from tokens

test['tokens']=test['tokens'].apply(lambda x: [item for item in x if item not in stop])
# Importing lemmatizer 

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

# Instantiating lemmatizer 

lemmatizer = WordNetLemmatizer()



lemmatize_words=[]

for i in range (len(test['tokens'])):

    word=''

    for j in range(len(test['tokens'][i])):

        lemm_word=lemmatizer.lemmatize(test['tokens'][i][j])#lemmatize

        

        word=word + ' '+lemm_word # joining tokens into sentence    

    lemmatize_words.append(word) # store in list



#creating a new column to store the result

test['lemmatized']=lemmatize_words

#displaying first 5 rows of dataframe

test.head()
# word cloud for Frequntly occuring words in test dataframe after lemmatizing

text=" ".join(post for post in test.lemmatized)

wordcloud = WordCloud(max_font_size=90, max_words=50, background_color="white", colormap="inferno").generate(text)

plt.figure(figsize=(10,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.title('\n Frequntly occuring words in test dataframe after lemmatizing \n\n',fontsize=18)

plt.axis("off")

plt.show()
#Text Vectorization using TfidfVectorizer //Convert a collection of raw documents to a matrix of TF-IDF features.

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

train_vectors = vectorizer.fit_transform(df["lemmatized"])

## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = vectorizer.transform(test["lemmatized"])
#imports

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
#defining X and y for the model

X = df['lemmatized']

y = df['target']
#Text Vectorization using TfidfVectorizer //Convert a collection of raw documents to a matrix of TF-IDF features.

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X)

X_test = vectorizer.transform(test["lemmatized"])
# Training the Logistic Regression model on the Training set

#from sklearn.linear_model import LogisticRegression

#classifier = LogisticRegression(C = 0.1)

#classifier.fit(X_train, y)
# Training the SVM model on the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(X_train, y)
from sklearn.metrics import classification_report

y_pred = classifier.predict(X_test)
# Creating an empty data frame

submission_kaggle = pd.DataFrame()

# Assigning values to the data frame-submission_kaggle

submission_kaggle['Id'] = test.id

submission_kaggle['target'] = y_pred

# Head of submission_kaggle

submission_kaggle.head()
# saving data as  final_kaggle.csv

submission_kaggle.loc[ :].to_csv('final_kaggle.csv',index=False)