# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')

df.head()
df = df.drop(['reviewerID','reviewerName'],axis=1)

df = df.rename(columns={'asin':'productID', 'overall':'rating', 'unixReviewTime':'unixTime'})

df.isnull().sum().sort_values(ascending=False)

df = df.dropna()



df.head()
df['reviewTime'] = pd.to_datetime(df['reviewTime'])

df['reviewYear'] = df['reviewTime'].dt.year

df = df.drop(['unixTime','reviewTime'],axis=1)



df.head()
def length(text):

    return len(text)



df['length'] = df['reviewText'].apply(length)



df.head()
df['sentiment'] = df['rating']

df['sentiment'] = df['sentiment'].replace({

    1:'negative',

    2:'negative',

    3:'neutral',

    4:'positive',

    5:'positive'

})



df.head()
def help_yes1(text):

    return text.split(',')[0]



def help_yes2(text):

    return text.split('[')[1]



def help_yesno1(text):

    return text.split(' ')[1]



def help_yesno2(text):

    return text.split(']')[0]



df['helpful_yes'] = df['helpful'].apply(help_yes1).apply(help_yes2).astype(int)

df['helpful_yesno'] = df['helpful'].apply(help_yesno1).apply(help_yesno2).astype(int)



df['helpful_pct'] = df['helpful_yes']/df['helpful_yesno']

df['helpful_pct'] = df['helpful_pct'].fillna(0)



df = df.drop(['helpful_yes','helpful_yesno'],axis=1)



df.head()
df = df[['productID','summary','reviewText','rating','sentiment','helpful','helpful_pct','length','reviewYear']]



df.head()
plt.figure(figsize=[10,5])

sns.countplot(df['rating'],palette='Wistia').set_title('Music Instrument Ratings Countplot')
import plotly.offline as py

import plotly.graph_objs as go



score = df['rating'].value_counts()



labels = score.index

values = score.values



scores = go.Pie(labels = labels,

               values = values,

               hole = 0.25)



df_scores = [scores]



layout = go.Layout(

           title = 'Percentage of Ratings for Amazon Musical Instruments')



fig = go.Figure(data = df_scores,

                 layout = layout)



py.iplot(fig)
df_h = df[df['helpful_pct'] > 0.75]
score_h = df_h['rating'].value_counts()



labels_h = score_h.index

values_h = score_h.values



scores_h = go.Pie(labels = labels_h,

               values = values_h,

               hole = 0.25)



df_scoresh = [scores_h]



layout_h = go.Layout(

           title = 'Percentage of Ratings for Amazon Musical Instruments (helpful reviews only)')



fig_h = go.Figure(data = df_scoresh,

                 layout = layout_h)



py.iplot(fig_h)
color = plt.cm.plasma(np.linspace(0, 1, 15))

df['productID'].value_counts()[:20].plot.bar(color = color, figsize = (10, 5))

plt.title('20 Most Reviewed Products', fontsize = 20)

plt.xlabel('Product ID')

plt.ylabel('Count')

plt.show()
year = df['reviewYear'].value_counts()



labels = year.index

values = year.values



years = go.Pie(labels = labels,

               values = values,

               hole = 0.25)



df_years = [years]



layout = go.Layout(

           title = 'Percentage of Years for Amazon Musical Instruments Review')



fig = go.Figure(data = df_years,

                 layout = layout)



py.iplot(fig)
plt.figure(figsize=[10,5])

sns.countplot(df['reviewYear'],hue=df['sentiment'],palette='Wistia')
product_rating = {}



for row, product in enumerate(df['productID'].unique()):

    product_temp = df[df['productID'] == product]

    product_rating[product] = product_temp['rating'].mean()
df_product_rating = pd.DataFrame(list(product_rating.items()), columns=['productID','rating'])



df_product_rating.head()
plt.figure(figsize=[10,5])

sns.distplot(df_product_rating['rating'],bins=35,kde=False).set_title('Distribution of Musical Instrument Ratings')
print(df_product_rating[df_product_rating['rating'] == df_product_rating['rating'].min()])
print(df[df['productID'] == 'B0025V1REU'])
df_bruh = df[(df['productID'] == 'B0025V1REU') & (df['sentiment'] == 'negative')]

df_bruh['reviewText'].iloc[0]
df_bruh['reviewText'].iloc[1]
df_bruh['reviewText'].iloc[2]
df['length'].plot(bins=50,kind='hist')
df[df['length'] == df['length'].max()]['reviewText'].iloc[0]
print('rating: ', df[df['length'] == df['length'].max()]['rating'].iloc[0])
print('helpful: ', df[df['length'] == df['length'].max()]['helpful'].iloc[0])
plt.figure(figsize=[10,5])

sns.violinplot(df['rating'], df['length'], palette='Wistia')

plt.title('Rating vs Length', fontsize = 20)
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer(stop_words = 'english')

words = cv.fit_transform(df['reviewText'])

sum_words = words.sum(axis=0)





words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



wordcloud = WordCloud().generate_from_frequencies(dict(words_freq))



plt.figure(figsize=(10, 10))

plt.axis('off')

plt.imshow(wordcloud)

plt.title("Most Common Words", fontsize = 20)

plt.show()
df_p = df.copy()
import string

from nltk.corpus import stopwords
def text_process(review):

    """

    Takes in a string of text, then perform the following:

    1. Remove all punctuations

    2. Remove all stopwords

    3. Return a list of cleaned text

    """

    # Check characters to see if they are in punctuation

    nopunc = [char for char in review if char not in string.punctuation]

    

    # Join the characters again to form the string

    nopunc = ''.join(nopunc)

    

    # Remove stopwords

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
df_p['reviewText'] = df_p['reviewText'].apply(text_process)
df_p['reviewText'].head()
from sklearn.model_selection import train_test_split



text_train, text_test, sent_train, sent_test = train_test_split(df_p['reviewText'], df_p['sentiment'], test_size=0.3)



print(len(text_train), len(text_test), len(sent_train) + len(sent_test))
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



pipeline = Pipeline([

    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(text_train,sent_train)
predictions = pipeline.predict(text_test)
from sklearn.metrics import classification_report

print(classification_report(predictions,sent_test))