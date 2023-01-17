!pip install comet_ml
# Loading in the comet_ml tool

#from comet_ml import Experiment

    

# Setting the API key, saved as environment variable

# experiment = Experiment(api_key="9gsTl4Wv73PDkYEoX8PUt5RSX",

#                       project_name="nlp-predict-first-class", workspace="ms-noxolo")

# experiment.display()
# Importing modules for data science and visualization

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 180

# NLP Libraries

import re

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import RegexpTokenizer

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

from nltk.stem import SnowballStemmer

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# ML Libraries

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from nltk import pos_tag

#ignore warnings

import warnings

warnings.filterwarnings('ignore')
!pip install nlppreprocess
# Loading in the datasets

train = pd.read_csv("/kaggle/input/climate-change-belief-analysis/train.csv")

test = pd.read_csv("/kaggle/input/climate-change-belief-analysis/test.csv")

sample_submission = pd.read_csv('/kaggle/input/climate-change-belief-analysis/sample_submission.csv')
# Looking at the first few entries in the dataset

train.head()
# Shape of the dataset

train.shape
# dataframe information

train.info()
# Looking at the numbers of possible classes in our sentiment

train['sentiment'].unique()
# Looking at the how the messages are distributed across the sentiment

train.describe()
# Checking for missing values

train.isnull().sum()
# Checking whether a character is white-space character or not

print(len(train['message']))

print(sum(train['message'].apply(lambda x: x.isspace())))
# Sample tweet

tweet = train.iloc[6,1]

print(tweet)
# Visualizing the distribution of the target 

plt.hist(train['sentiment'], label='data');

plt.legend();

plt.title('Distribution of target labels')
# Distribution plots for the label

fig,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(16,8))



#For Positive 

sns.distplot(train[train['sentiment']==1]['message'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'blue', 

             ax = ax1,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax1.set_title('Positive')

ax1.set_xlabel('message_Length')

ax1.set_ylabel('Density')



#For Negative 

sns.distplot(train[train['sentiment']==-1]['message'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'lightblue', 

             ax = ax2,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax2.set_title('Negative ')

ax2.set_xlabel('message_Length')

ax2.set_ylabel('Density')



#For Neutral 

sns.distplot(train[train['sentiment']==0]['message'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'purple',  

             ax = ax3,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax3.set_title('Neutral ')

ax3.set_xlabel('message_Length')

ax3.set_ylabel('Density')



#For Neews

sns.distplot(train[train['sentiment']==2]['message'].str.len(), hist=True, kde=True,

             bins=int(200/25), color = 'green', 

             ax = ax4,

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax4.set_title('News')

ax4.set_xlabel('message_Length')

ax4.set_ylabel('Density')
working_df = train.copy()

# Labeling the target

working_df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in working_df['sentiment']]
# checking the numerical distribution

values = working_df['sentiment'].value_counts()/working_df.shape[0]

labels = (working_df['sentiment'].value_counts()/working_df.shape[0]).index

colors = ['lightgreen', 'blue', 'purple', 'lightsteelblue']

plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0, 0, 0), colors=colors)

plt.show()
sns.countplot(x='sentiment' ,data = working_df, palette='PRGn')

plt.ylabel('Count')

plt.xlabel('Sentiment')

plt.title('Number of Messages Per Sentiment')

plt.show()
# Visualizing text lengths for each sentiment

sns.barplot(x='sentiment', y=working_df['message'].apply(len) ,data = working_df, palette='PRGn')

plt.ylabel('Length')

plt.xlabel('Sentiment')

plt.title('Average Length of Message by Sentiment')

plt.show()
# Extracting Users in a column

working_df['users'] = [''.join(re.findall(r'@\w{,}', line)) 

                       if '@' in line else np.nan for line in working_df.message]
# Generating Counts of users

counts = working_df[['message', 'users']].groupby('users', as_index=False).count().sort_values(by='message', ascending=False)
# Top 5 most popular

counts.head()
# checking the numerical distribution

values = [sum(np.array(counts['message']) == 1)/len(counts['message']), sum(np.array(counts['message']) != 1)/len(counts['message'])]

labels = ['First Time Tags', 'Repeated Tags']

colors = ['lightsteelblue', "purple"]

plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=90, explode= (0.04, 0), colors=colors)

plt.show()
# Analysis of most popular tags, sorted by populariy

sns.countplot(y="users", hue="sentiment", data=working_df, palette='PRGn',

              order=working_df.users.value_counts().iloc[:20].index) 



plt.ylabel('User')

plt.xlabel('Number of Tags')

plt.title('Top 20 Most Popular Tags')

plt.show()

#plt.xticks(rotation=90)
# Analysis of most popular tags, sorted by populariy

sns.countplot(x="users", data=working_df[working_df['sentiment'] == 'Positive'],

              order=working_df[working_df['sentiment'] == 'Positive'].users.value_counts().iloc[:20].index) 



plt.xlabel('User')

plt.ylabel('Number of Tags')

plt.title('Top 20 Positive Tags')

plt.xticks(rotation=85)

plt.show()
# Analysis of most popular tags, sorted by populariy

sns.countplot(x="users", data=working_df[working_df['sentiment'] == 'Negative'],

              order=working_df[working_df['sentiment'] == 'Negative'].users.value_counts().iloc[:20].index) 



plt.xlabel('User')

plt.ylabel('Number of Tags')

plt.title('Top 20 Negative Tags')

plt.xticks(rotation=85)

plt.show()
# Analysis of most popular tags, sorted by populariy

sns.countplot(x="users", data=working_df[working_df['sentiment'] == 'News'],

              order=working_df[working_df['sentiment'] == 'News'].users.value_counts().iloc[:20].index) 



plt.xlabel('User')

plt.ylabel('Number of Tags')

plt.title('Top 20 News Tags')

plt.xticks(rotation=85)

plt.show()
# Testing the PorterStemmer 

stemmer = PorterStemmer()

print("The stemmed form of typing is: {}".format(stemmer.stem("typing")))

print("The stemmed form of types is: {}".format(stemmer.stem("types")))

print("The stemmed form of type is: {}".format(stemmer.stem("type")))
# Testing Lemmatization

lemm = WordNetLemmatizer()

print("In  case of Lemmatization, typing is: {}".format(lemm.lemmatize("typing")))

print("In  case of Lemmatization, types is: {}".format(lemm.lemmatize("types")))

print("In  case of Lemmatization, type is: {}".format(lemm.lemmatize("type")))
from nlppreprocess import NLP

nlp = NLP()

nlp.process('shouldnt')
nlp.process('There is no good here')
# Data cleaning for furthur sentiment analysis



def cleaner(line):

    '''

    For preprocessing the data, we regularize, transform each upper case into lower case, tokenize,

    normalize and remove stopwords. Normalization transforms a token to its root word i.e. 

    These words would be transformed from "love loving loved" to "love love love."

    

    '''



    # Removes RT, url and trailing white spaces

    line = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', line).strip()) 

    emojis = re.compile("["

                           u"\U0001F600-\U0001F64F"  # removes emoticons,

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)



    line = emojis.sub(r'', line)



    # Removes puctuation

    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")

    tweet = punctuation.sub("", line.lower()) 

    # Removes stopwords

    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, 

                            remove_numbers=True, remove_punctuations=False) 

    tweet = nlp_for_stopwords.process(tweet) # This will remove stops words that are not necessary. The idea is to keep words like [is, not, was]

    # https://towardsdatascience.com/why-you-should-avoid-removing-stopwords-aa7a353d2a52



    # tokenisation

    # We used the split method instead of the word_tokenise library because our tweet is already clean at this point

    # and the twitter data is not complicated

    tweet = tweet.split() 

    # POS 

    # Part of Speech tagging is essential for Lemmatization to perform well

    pos = pos_tag(tweet)

    

    # Lemmatization

    lemmatizer = WordNetLemmatizer()

    tweet = ' '.join([lemmatizer.lemmatize(word, po[0].lower()) 

                      if (po[0].lower() in ['n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])

    return tweet
text1 = cleaner(tweet)

print('BEFORE')

print(tweet, '\n'*2)

print('AFTER')

print(text1)

# In the tweet below, you can see that "not" was added and kept, because the word is significant
cleaned = train['message'].apply(cleaner)
cleaned.head()
working_df['clean'] = cleaned
# Combining all the messages

text_before_cleaning = " ".join(tweet for tweet in train['message'])

text_after_cleaning = " ".join(tweet for tweet in cleaned)
# Numbers of characters

sns.barplot(x=['Before Cleaning', 'After Cleaning'], y=[len(text_before_cleaning), len(text_after_cleaning)], palette='PRGn')

# sns.countplot(x=[] ,data = working_df, palette='PRGn')

plt.ylabel('Number of Characters')

# plt.xlabel('Sentiment')

plt.title('Number of Characters')

plt.show()
# Generating the word cloud image from all the messages

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(text_after_cleaning)



# Displaying the word cloud image:

# using matplotlib way:

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Wordcloud for the cleaned data:Positive sentiment

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['clean'][working_df['sentiment'] == 'Positive']))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud using matplotlib:

# plt.title("General Word Cloud")

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Wordcloud for the cleaned data:Negative sentiment

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['clean'][working_df['sentiment'] == 'Negative']))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud image:

# plt.title("General Word Cloud")

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Wordcloud for the cleaned data:News

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['clean'][working_df['sentiment'] == 'News']))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud image:

# plt.title("General Word Cloud")

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Creating a column of hastags

working_df['hashtags'] = [' '.join(re.findall(r'#\w{,}', line)) 

                       if '#' in line else np.nan for line in working_df.message]
# A sneak peak of these tweets plus hashtags

working_df.head()
# Creating a wordcloud of the hashtags: Positive sentiment

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['hashtags'][working_df['sentiment'] == 'Positive'] if type(tweet) == str))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud image:

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Wordcloud for the hashtags:Negtive sentiment

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['hashtags'][working_df['sentiment'] == 'Negative'] if type(tweet) == str))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud image:

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Wordcloud for the hashtags:News

corpus = re.sub("climate change", ''," ".join(tweet.strip() for tweet in working_df['hashtags'][working_df['sentiment'] == 'News'] if type(tweet) == str))

wordcloud = WordCloud(font_path='../input/droidsansmonottf/droidsansmono.ttf', background_color="white",

                      width = 1920, height = 1080, colormap="viridis").generate(corpus)



# Displaying the word cloud image:

# plt.title("General Word Cloud")

plt.figure(dpi=260)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
# Converting the collection of clean messages to a matrix of token counts

corpus = working_df['clean']



vectorizer = CountVectorizer()

count_vectorized = vectorizer.fit_transform(corpus)

#print(vectorizer.get_feature_names())

#print(X.toarray())
# Converting the collection of clean messages to a matrix of TF-IDF features

data = working_df['clean']



vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)

vectorized = vectorizer.fit_transform(data)

#print(vectorizer.get_feature_names())

#print(X.toarray())
# Using sparse to train the model using both representations.

import scipy.sparse



# Defining the features as well as the label

X = scipy.sparse.hstack([vectorized, count_vectorized])

y = working_df['sentiment']
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.decomposition import LatentDirichletAllocation as LDA

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.model_selection import train_test_split



# Splitting the previously defined features and label of your dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Creating a list of all the models to train

algs = [LogisticRegression(random_state = 5), SVC(kernel = 'linear', random_state = 5), SVC(kernel = 'rbf', random_state = 5)

        ,MultinomialNB(), KNeighborsClassifier(), DecisionTreeClassifier(max_depth=6),RandomForestClassifier()]
# Fitting models onto the training data and predicting.

for i in range(0, len(algs)):

    text_clf = Pipeline([('clf', algs[i])])

    ##lowercase = True,stop_words='english', ngram_range=(1, 2), analyzer='word',max_df = 0.8

    text_clf.fit(X_train, y_train)  

    predictions = text_clf.predict(X_test)

    

    

    print(algs[i])

    print(metrics.confusion_matrix(y_test,predictions))

    print(metrics.classification_report(y_test,predictions))

    print('F1_score: ',round(metrics.f1_score(y_test,predictions, average = 'weighted'),3))

    print('-------------------------------------------------------')
data = train.copy()
# importing the module and creating a resampling variable

from sklearn.utils import resample

class_size = int(len(data[data['sentiment']==1])/2)
# seperating the four classes

class_1 = data[data['sentiment']==-1]

class_2 = data[data['sentiment']==0]

class_3 = data[data['sentiment']==1]

class_4 = data[data['sentiment']==2]
# upsampling classes 1, 2, and 4 & downsampling class 3

class_1_up = resample(class_1,replace=True,n_samples=class_size, random_state=27)

class_2_up = resample(class_2,replace=True,n_samples=class_size, random_state=27)

class_4_up = resample(class_4,replace=True,n_samples=class_size, random_state=27)

class_3_down = resample(class_3,replace=False,n_samples=class_size, random_state=27)
# Creating a new DataFrame out of the balanced bata

res_df = pd.concat([class_1_up, class_2_up, class_4_up,class_3_down])
# Checking if data has been well-balanced

sns.countplot(x = res_df['sentiment'], data = data, palette='PRGn')

plt.show()
# Defining the features as well as the label

X1 = res_df['message']

X_res = X1.apply(cleaner)

y_res = res_df['sentiment']



X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)
# Fitting models onto the training data and predicting.

for i in range(0, len(algs)):

    text_clf = Pipeline([('count_vec', CountVectorizer(lowercase = True, ngram_range=(1, 2), analyzer='word')),

                         ('clf', algs[i]),])

    text_clf.fit(X_train, y_train)  

    predictions = text_clf.predict(X_test)

    

    

    print(algs[i])

    print(metrics.confusion_matrix(y_test,predictions))

    print(metrics.classification_report(y_test,predictions))

    print('F1_score: ',round(metrics.f1_score(y_test,predictions, average = 'weighted'),3))

    print('-------------------------------------------------------')
# Extracting a confusion matrix from the results of balanced data

from sklearn import metrics

print(metrics.confusion_matrix(y_test, predictions))
# Extracting a classification report from the balanced data

print(metrics.classification_report(y_test, predictions))