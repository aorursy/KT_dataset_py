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
df = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')

df1 = pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')
df.shape, df1.shape
df.describe(include = 'all')
df1.isnull().sum()[df1.isnull().sum()>0]
df1 = df1.dropna()
df1['Rating'] =df1['Rating'].str.replace('Like', '5')
df1['Rating'] = df1['Rating'].astype(float)
df1['Time'] = pd.to_datetime(df1['Time'])

df1['Time'].min(),df1['Time'].max()
df1['Metadata'] =df1['Metadata'].str.replace(' Review', ' Reviews')
df1['reviews'] = df1['Metadata'].str.replace('[^0-9,]','').str.split(',').str[0].astype(float)

df1['followers'] = df1['Metadata'].str.replace('[^0-9,]','').str.split(',').str[1].astype(float)
df1['reviews'] = df1['reviews'].astype(float)
df1['followers'].fillna('0', inplace = True)
df1['followers'] = df1['followers'].astype(float)
df1['Time'] = pd.to_datetime(df1['Time'])

df1['Day'] = df1['Time'].dt.day

df1['Month'] = df1['Time'].dt.month

df1['Year'] = df1['Time'].dt.year
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15, 8))

res_rating_5 = df1.groupby(['Restaurant','Rating'])['Rating'].count()

top_res_having_5_ratings = res_rating_5.sort_values(ascending = False).head(11)

chart1 = top_res_having_5_ratings[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Number_of_5_Ratings')

plt.xlabel('Restaurant_Name')
plt.figure(figsize=(15, 4))

res_max_pics = df1.groupby('Restaurant')['Pictures'].max()

res_with_more_pics = res_max_pics.sort_values(ascending = False).head(21)

chart1 = res_with_more_pics[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Pictures_Count')

plt.xlabel('Restaurant_Name')
plt.figure(figsize=(20, 8))

chart1 = sns.countplot(x = 'Reviewer', data=df1,

              order=df1.Reviewer.value_counts().iloc[:10].index)

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')
plt.figure(figsize=(15, 4))

res_avg_rating = df1.groupby('Restaurant')['Rating'].mean()

top10_res = res_avg_rating.sort_values(ascending = False).head(10)

chart1 = top10_res[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('Restaurant_Name')

plt.ylabel('Avg_Rating')
plt.figure(figsize=(25, 6))

reviewers_total_pics_posted = df1.groupby('Reviewer')['Pictures'].sum()

reviewers_with_30_or_more_pics = reviewers_total_pics_posted.sort_values(ascending = False).head(33)

chart1 = reviewers_with_30_or_more_pics[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Total Number of Pictures Posted by Reviewer')

plt.xlabel('Reviewer_Name')
plt.figure(figsize=(15, 6))

total_reviews_of_reviewers = df1.groupby('Reviewer')['reviews'].sum()

top10_reviewers = total_reviews_of_reviewers.sort_values(ascending = False).head(10)

chart1 = top10_reviewers[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('Reviewers_Name')

plt.ylabel('Total_Views_on-the_Reviews_by_Reviewers')
plt.figure(figsize=(15, 6))

total_followers_of_reviewers = df1.groupby('Reviewer')['followers'].sum()

top10_reviewers = total_followers_of_reviewers.sort_values(ascending = False).head(10)

chart1 = top10_reviewers[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.xlabel('Reviewers_Name')

plt.ylabel('Total_Followers_of_the_Reviewers')
plt.figure(figsize=(15, 4))

res_avg_rating = df1.groupby(['Restaurant', 'Year'])['Rating'].mean()

top10_res = res_avg_rating.sort_values(ascending = False).head(10)

chart1 = top10_res[::-1].plot.bar()

for p in chart1.patches:

    chart1.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., 

                                                   p.get_height()), ha = 'center', va = 'center', 

                   xytext = (0, 10), textcoords = 'offset points')

plt.ylabel('Rating')

plt.xlabel('Restaurant_Name')
plt.figure(figsize=(15, 4))

df1.resample('1D',on='Time')['Restaurant'].size().plot.line() ## Instead of '1Y' we can use '1d' or '1m' or '1H'

plt.xlabel('Date')

plt.ylabel('No. of Reviews')

plt.show()
from wordcloud import WordCloud



plt.figure(figsize=(15, 4))

ip_string = ' '.join(df1['Review'].dropna().to_list())



wc = WordCloud(background_color='white').generate(ip_string.lower())

plt.imshow(wc)
# Word_Count

df1['Word_Count'] = df1['Review'].apply(lambda x: len(str(x).split()))
# Character_Count

df1['Char_Count'] = df1['Review'].apply(lambda x: len(x))
# Count hashtags(#) and @ mentions



df1['hashtags_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.startswith('#')]))

df1['mention_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
# If numeric digits are present in tweets



df1['numerics_count'] = df1['Review'].apply(lambda x: len([t for t in x.split() if t.isdigit()]))
# UPPER_case_words_count#



df1['UPPER_CASE_COUNT'] = df1['Review'].apply(lambda x: len([t for t in  x.split()

                                                             if t.isupper() and len(x)>3]))
import re
# Count and Removing Emails

df1['Emails'] = df1['Review'].apply(lambda x: re.findall(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)',x))

df1['Review'] = df1['Review'].apply(lambda x: re.sub(r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)', '',x))
# Count and Remove URL's

df1['URL_Flags'] = df1['Review'].apply(lambda x: len(re.findall(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', x)))

df1['Review'] = df1['Review'].apply(lambda x: re.sub(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', x))
# Removing RE_REVIEWS

df1['Review'] = df1['Review'].apply(lambda x: re.sub('RT', '', x))
# Punctuation_Count

df1['punct_count'] = df1['Review'].apply(lambda x: len(re.findall('[^a-z A-Z 0-9-]+', x)))
# Removal of special chars and punctuation

df1['Review'] = df1['Review'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', x))
# Removing_Multiple_Spaces

df1['Review'] = df1['Review'].apply(lambda x: ' '.join(x.split()))
# Preprocessing and cleaning



contractions = {

"aight": "alright",

"ain't": "am not",

"amn't": "am not",

"aren't": "are not",

"can't": "can not",

"cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"daren't": "dare not",

"daresn't": "dare not",

"dasn't": "dare not",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"d'ye": "do you",

"e'er": "ever",

"everybody's": "everybody is",

"everyone's": "everyone is",

"finna": "fixing to",

"g'day": "good day",

"gimme": "give me",

"giv'n": "given",

"gonna": "going to",

"gon't": "go not",

"gotta": "got to",

"hadn't": "had not",

"had've": "had have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he would",

"he'dn't've'd": "he would not have had",

"he'll": "he will",

"he's": "he is",

"he've": "he have",

"how'd": "how did",

"howdy": "how do you do",

"how'll": "how will",

"how're": "how are",

"I'll": "I will",

"I'm": "I am",

"I'm'a": "I am about to",

"I'm'o": "I am going to",

"innit": "is it not",

"I've": "I have",

"isn't": "is not",

"it'd": "it would",

"it'll": "it will",

"it's": "it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"may've": "may have",

"methinks": "me thinks",

"mightn't": "might not",

"might've": "might have",

"mustn't": "must not",

"mustn't've": "must not have",

"must've": "must have",

"needn't": "need not",

"ne'er": "never",

"o'clock": "of the clock",

"o'er": "over",

"ol'": "old",

"oughtn't": "ought not",

"'s": "is, has, does, or us",

"shalln't": "shall not",

"shan't": "shall not",

"she'd": "she had",

"she'll": "she will",

"she's": "she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"somebody's": "somebody is",

"someone's": "someone is",

"something's": "something is",

"so're": "so are",

"that'll": "that will",

"that're": "that are",

"that's": "that is",

"that'd": "that would",

"there'd": "there had",

"there'll": "there will",

"there're": "there are",

"there's": "there is",

"these're": "these are",

"these've": "these have",

"they'd": "they had",

"they'll": "they will",

"they're": "they are",

"they've": "they have",

"this's": "this is",

"those're": "those are",

"those've": "those have",

"'tis": "it is",

"to've": "to have",

"'twas": "it was",

"wanna": "want to",

"wasn't": "was not",

"we'd": "we had",

"we'll": "we will",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'd": "what did",

"what'll": "what will",

"what're": "what are",

"what's": "what is",

"what've": "what have",

"when's": "when is",

"where'd": "where did",

"where'll": "where will",

"where're": "where are",

"where's": "where is",

"where's": "where does",

"where've": "where have",

"which'd": "which would",

"which'll": "which will",

"which're": "which are",

"which's": "which is",

"which've": "which have",

"who'd": "who would",

"who'd've": "who would have",

"who'll": "who will",

"who're": "who are",

"who's": "who does",

"who've": "who have",

"why'd": "why did",

"why're": "why are",

"why's": "why does",

"won't": "will not",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd've": "you all would have",

"y'all'dn't've'd": "you all would not have had",

"y'all're": "you all are",

"you'd": "you would",

"you'll": "you will",

"you're": "you are",

"you've": "you have",

" u ": "you",

" ur ": "your",

" n ": "and"

}
def cont_to_exp(x):

    if type(x) is str:

        for key in contractions:

            value = contractions[key]

            x = x.replace(key,value)

        return x

    else:

        return x
df1['Review'] = df1['Review'].apply(lambda x: cont_to_exp(x))
from textblob import TextBlob



pol = lambda x: TextBlob(x).sentiment.polarity

sub = lambda x: TextBlob(x).sentiment.subjectivity



df1['polarity'] = df1['Review'].apply(pol)

df1['subjectivity'] = df1['Review'].apply(sub)

df1.head()
df1['Sentiments'] = df1['polarity'].apply(lambda v: 'Positive' if v>0.000000 else ('Negative' if v<0.000000 else 'Neutral'))
import spacy

nlp = spacy.load('en_core_web_sm')

import string
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
punct = string.punctuation
def text_data_cleaning(sentence):

    doc = nlp(sentence)

    tokens = []

    for token in doc:

        if token.lemma_ != '-PRON-':

            temp = token.lemma_.lower().strip()

        else:

            temp = token.lower_

        tokens.append(temp)

    

    cleaned_tokens = []

    for token in tokens:

        if token not in stopwords and token not in punct:

            cleaned_tokens.append(token)

    return cleaned_tokens
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline



from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.preprocessing import MinMaxScaler
tfidf = TfidfVectorizer(tokenizer = text_data_cleaning)

classifier = LinearSVC()
X = df1['Review']

y = df1['Sentiments']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)