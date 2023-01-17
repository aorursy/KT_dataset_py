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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#loading the data set : 

train_df = pd.read_csv("../input/nlp-getting-started/train.csv")

test_df = pd.read_csv("../input/nlp-getting-started/test.csv")
train_df.head(5)
train_df.info()
train_df.shape
train_df.columns
# data visualization

sns.countplot('target', data=train_df)
train_df['keyword'].value_counts()[:100]
#plot ref : https://mode.com/python-tutorial/counting-and-plotting-in-python/

plt.figure(figsize=(10,5))

train_df['keyword'].value_counts()[:20].plot(kind='bar')
#plotting ref : https://www.kaggle.com/tejainece/seaborn-barplot-and-pandas-value-counts



keyword_count  = train_df['keyword'].value_counts()

keyword_count = keyword_count[:20,]

plt.figure(figsize=(20,5))

sns.barplot(keyword_count.index, keyword_count.values, alpha=0.8)

plt.title('Top 10 keywords')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('keywords', fontsize=12)

plt.show()



location_count  = train_df['location'].value_counts()

location_count = location_count[:20,]

plt.figure(figsize=(25,5))

sns.barplot(location_count.index, location_count.values, alpha=0.8)

plt.title('Top 10 locations')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('location', fontsize=12)

plt.show()
# what i need to do is ... find out the length of the text for each row and make a cloumn out 

#of it and add to the existing dataframe. try to see any correlation between 

# the length of text and target.. 



train_df['Length_of_text'] = train_df['text'].apply(lambda x : len(x))

train_df.head(4)

sns.distplot(train_df['Length_of_text'])
train_df['target'].value_counts()
sns.distplot(train_df[(train_df['target'] == 1)]['Length_of_text'])

sns.distplot(train_df[(train_df['target'] == 0)]['Length_of_text'])
train_df.Length_of_text.describe()
train_df.hist(column='Length_of_text', by='target', bins=50, figsize=(10,5))
#references : https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/

#https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f

#https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/

#https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908

#https://www.kaggle.com/saxinou/nlp-01-preprocessing-data
train_df['location'].value_counts()
# EDA analysis : 

"""

Ref : https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c

      https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb

      https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/

"""



#train_df.hist(column = 'Length_of_text', by = 'target', bins = 50, figsize = (10,5))

#train_df['Length_of_text'].describe()



#ref : https://towardsdatascience.com/the-real-world-as-seen-on-twitter-sentiment-analysis-part-one-5ac2d06b63fb

plt.figure(figsize=(10,5))

x = train_df['Length_of_text'][train_df.target == 1]

y = train_df['Length_of_text'][train_df.target == 0]

g = plt.hist([x, y], color=['r','b'], alpha=0.5, label=['positive','negative'])

plt.legend(loc='upper left')
#need to find the distribution of the target

#i.e how many 0 and 1.. 

print(train_df['target'].value_counts())

#train_df['target'].plot(kind='pie')

values = [4342, 3271]

colors = ['r', 'g']

labels = ['0', '1']

plt.figure(figsize=(7,7))

plt.pie(values, colors=colors, labels= values, autopct='%1.1f%%')

plt.title('Distribution of real (1) and fake (0) disaster')

plt.legend(labels,loc=3)

plt.show()
#print(train_df['keyword'].value_counts())

# try to construct a dataframe out of the value counts and plot a pie chart to see which are the most common disaster.. repeat the same for location



df_keywords = pd.DataFrame(train_df['keyword'].value_counts(ascending=True))

df_keywords = df_keywords.reset_index()

df_keywords.columns = ['Keywords', 'No of occurances']



df_keywords.head(10)



plt.figure(figsize=(10,50))

plt.grid(color = 'red', linestyle = '--')

plt.margins(y=0)

plt.barh(df_keywords['Keywords'], df_keywords['No of occurances'], alpha = 0.5,  align = 'edge', color = sns.color_palette('Paired'))



#plt.figure(figsize=(7,7))

#plt.pie(df_keywords['No of occurances'], colors=colors, labels= df_keywords['Keywords'], autopct='%1.1f%%')

#plt.title('Distribution of disasters')

#plt.legend(labels,loc=3)

#plt.show()



#df_keywords.plot(kind = 'barh')
df_location = pd.DataFrame(train_df['location'].value_counts(ascending=True))

df_location = df_location.reset_index()

df_location.columns = ['Location', 'No of occurances']



df_location.head(20)



#df_location[df_location['No of occurances']!=1].plot(kind = 'barh')

#df_location[df_location['No of occurances']!=1]['Location']



plt.figure(figsize=(10,100))

plt.grid(color = 'red', linestyle = '--')

plt.margins(y=0)

plt.barh(df_location[df_location['No of occurances']>1]['Location'], df_location[df_location['No of occurances']>1]['No of occurances'], alpha = 0.5,  align = 'edge', color = sns.color_palette('Paired')) 
#print(train_df['keyword'].value_counts())

# try to construct a dataframe out of the value counts and plot a pie chart to see which are the most common disaster.. repeat the same for location



#train_df.head(30)

#print(train_df[10:40])

#train_df[train_df['target'] == 1]['keyword'].value_counts(ascending=True)

#train_df['keyword'].value_counts(ascending=True)



df_keywords = pd.DataFrame(train_df[train_df['target'] == 0]['keyword'].value_counts(ascending=True))

df_keywords = df_keywords.reset_index()

df_keywords.columns = ['Keywords', 'No of occurances']



df_keywords.head(10)







plt.figure(figsize=(10,50))

plt.title("Distribution for real cases")

plt.grid(color = 'red', linestyle = '--')

plt.margins(y=0)

plt.barh(df_keywords['Keywords'], df_keywords['No of occurances'], alpha = 0.5,  align = 'edge', color = sns.color_palette('Paired'))



plt.figure(figsize=(50,10))

plt.grid(color = 'red', linestyle = '--')

plt.margins(y=0)

sns.countplot(x = 'Length_of_text', data = train_df, hue = 'target')
plt.figure(figsize=(10,80))

plt.grid(color = 'red', linestyle = '--')

plt.margins(x=0)

sns.countplot(y = 'keyword', data = train_df, hue = 'target')
# unique words ,,, how is it effecting the result 

train_df['text'].unique()
# TEXT Preprocessing : 

#ref : https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f 

""""

Text preprocessing is required to clean the text, to discard the not so important material.. 

These are the steps. 

1. Remove html

2. Tokenize

3. Remove punctuation and blank spaces if seems unneccesary

4. Remove stop words

5. Lemmatize and Stemming



"""

#printed a particular txt

print(train_df[44:45])

text1 = train_df['text'][44]

print("\n Actual text \n", text1)



# tokenize i.e. get the words out of the sentence

#words = train_df['text'][3].split()

#print(words)



# one can also use nltk for tokenizing

# ref : https://www.kaggle.com/saxinou/nlp-01-preprocessing-data





import re 

text1 = re.sub(r"http\S+", "", text1)

print("\n Text after removing url \n", text1)



import string

nopunc = [char for char in text1 if char not in string.punctuation] # removing punctuation

print("\n Text after applying .punctuation \n", nopunc)

text1 = "".join(nopunc)



import nltk

#tokens = nltk.word_tokenize(text1) # tokenize

#tokens = [t.lower() for t in tokens] # having same cases...i.e. either upper or lower -----> Normalization



tokens = nltk.word_tokenize(text1.lower()) # performing Normaliztion and tokenization together 

tokens = [t.strip() for t in tokens] # removing blank spaces



from nltk.corpus import stopwords

tokens = [w for w in tokens if w not in stopwords.words('english')]



print("\nTokenizing text into words With NLTK \n", tokens)



#import lemmatizer  and stemmer : 

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

lm = WordNetLemmatizer()

ps = PorterStemmer()



# 1- Stemming

words_stem = [ps.stem(w) for w in tokens]

print(type(words_stem))

print(words_stem)

    

# 2- Lemmatization

clean_tokens = [lm.lemmatize(w) for w in words_stem]

print(type(clean_tokens))

print(clean_tokens)
# clean data for the whole dataframe



#printed a particular txt

#print(train_df[44:45])

#text1 = train_df['text'][44]





def cleaning_process(text1) :    

#text11 = train_df['text'][44]printnt("\n Actual text \n", text1)

 text = text1

 text = re.sub(r"http\S+", "", text) # removing url

 nopunc = [char for char in text if char not in string.punctuation] # removing punctuation

 text = "".join(nopunc)

 tokens = nltk.word_tokenize(text.lower()) # performing Normaliztion and tokenization together 

 tokens = [t.strip() for t in tokens] # removing blank spaces

 tokens = [w for w in tokens if w not in stopwords.words('english')]

 lm = WordNetLemmatizer()

 ps = PorterStemmer()

 words_stem = [ps.stem(w) for w in tokens] # stemming

 clean_tokens = [lm.lemmatize(w) for w in words_stem] # lemmatizing   

 return clean_tokens





clean_text = train_df['text'].apply(lambda x : cleaning_process(x))

print(train_df['text'][30:40])

print(clean_text[30:40])



train_df['cleaned_text'] = clean_text

train_df['cleaned_text'] = train_df['cleaned_text'].apply(lambda x : ' '.join(x))



train_df.head(4)
#visualizing all the words in column "tweet_stemmed" in our data using the wordcloud plot.

all_words = ' '.join([text for text in train_df['cleaned_text']])

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)



plt.figure(figsize=(10, 7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis('off')

plt.title("Most Common words")

plt.show()



# vectorization

"""

some good reads : 

https://www.analyticsvidhya.com/blog/2018/07/hands-on-sentiment-analysis-datasehttps://www.kaggle.com/pceccon/countvectorizer-and-tfidf-strategiest-python/

https://www.kaggle.com/mistryjimit26/twitter-sentiment-analysis-basic

https://www.kaggle.com/gauravchhabra/nlp-twitter-sentiment-analysis-project ----- took guidance from this

https://www.kaggle.com/alvations/basic-nlp-with-nltk

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/46073

https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle --- have used grid search -- would like to implement

"""



# for vectorization one can use Countvectorizer or TfIdf... we will use both and try to compare 



# Bag of Words : -- CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english')

#cv

bow = cv.fit_transform(train_df['cleaned_text'])

#print(cv.get_feature_names())

#print(bow.shape)





# now let us also use Tf-Idf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

bow_tfidf = tfidf.fit_transform(train_df['cleaned_text'])

#print(tfidf.get_feature_names())

#print(bow_tfidf.shape)



print(type(bow_tfidf))

# modelling and train test split

# https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle

# https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/



from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(bow_tfidf, train_df['target'], test_size=0.2, random_state = 0)





from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score



model_MNB = MultinomialNB()

model_MNB.fit(msg_train, label_train) # training the model



predictions = model_MNB.predict(msg_test) # predicting on the validation set





from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(" Multinomial NB with TFIDF")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))
from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(bow, train_df['target'], test_size=0.2, random_state = 0)





from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score



model_MNB = MultinomialNB()

model_MNB.fit(msg_train, label_train) # training the model



predictions = model_MNB.predict(msg_test) # predicting on the validation set





from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(" Multinomial NB with CountVectorizer")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))
# using pipeline for training data 

#https://www.kaggle.com/youben/twitter-sentiment-analysis 

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

transform = TfidfTransformer()



pipeline = Pipeline([

    ('cv', CountVectorizer(stop_words='english')),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

    ])



msg_train, msg_test, label_train, label_test = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state = 0)

#msg_train, msg_test, label_train, label_test = train_test_split(train_df['cleaned_text'], train_df['target'], test_size=0.2, random_state = 0)

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)



print(" Multinomial NB with CountVectorizer using Pipeline")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression



pipeline = Pipeline([

    ('cv', CountVectorizer(stop_words='english')),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Naive Bayes classifier

    ])



#msg_train, msg_test, label_train, label_test = train_test_split(train_df['text'], train_df['target'], test_size=0.2, random_state = 0)

msg_train, msg_test, label_train, label_test = train_test_split(train_df['cleaned_text'], train_df['target'], test_size=0.2, random_state = 0)

pipeline.fit(msg_train,label_train)

predictions = pipeline.predict(msg_test)



print(" Logistic Regression with CountVectorizer using Pipeline")

print(confusion_matrix(label_test,predictions))

print(classification_report(label_test,predictions))

print(accuracy_score(label_test, predictions))

# now the test data



print('Head of Test data frame \n', test_df.head(4))

print('\n Info of Test data frame')

print(test_df.info())

print('\n Describe of Test data frame \n', test_df.describe())
# predicting for the test data



test_df['cleaned_text'] = test_df['text'].apply(lambda x : cleaning_process(x))

test_df['cleaned_text'] = test_df['cleaned_text'].apply(lambda x : ' '.join(x))



#test_df['target_predictions'] = pipeline.predict(test_df['text'])

test_df['target'] = pipeline.predict(test_df['cleaned_text']) # this is the  prediction

test_df.head(30)
sub_df = test_df[['id', 'target']]

sub_df.head(4)

sub_df.to_csv('submission.csv',index=False)

#! more submission.csv