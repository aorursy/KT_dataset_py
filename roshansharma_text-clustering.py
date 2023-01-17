# for importing dataset

import numpy as np

import pandas as pd

    

# for performing text clustering    

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.metrics import adjusted_rand_score



# for providing the path

import os

print(os.listdir('../input/'))



# for visualization

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')
# importing the excel file

data = pd.read_excel('../input/Text Clustering.xlsx', header = None)



# converting the data  to csv format

data.to_csv('your_csv.csv', encoding='utf-8')



# reading the data in csv format

data = pd.read_csv('your_csv.csv')



# getting the shape

data.shape
# assigning names to the columns



data.columns = ['Id', 'Text']



# checking the names of the columns

data.columns
# getting the length of the text as another feature



data['Length'] = data['Text'].apply(len)
# describing the dataset



data.groupby('Length').describe().head(20)
# looking at the head of the data



data.head()
# looking at the distribution of length of the different texts



plt.rcParams['figure.figsize'] = (15, 7)

sns.distplot(data['Length'], color = 'purple')

plt.title('The Distribution of Length over the Texts', fontsize = 20)
# wordcloud



from wordcloud import WordCloud



wordcloud = WordCloud(background_color = 'lightcyan',

                      width = 1200,

                      height = 700).generate(str(data['Text']))



plt.figure(figsize = (15, 10))

plt.imshow(wordcloud)

plt.title("WordCloud ", fontsize = 20)
from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer(stop_words = 'english')

words = cv.fit_transform(data['Text'])

sum_words = words.sum(axis=0)





words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



color = plt.cm.twilight(np.linspace(0, 1, 20))

frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = color)

plt.title("Most Frequently Occuring Words - Top 20")
from sklearn.feature_extraction.text import CountVectorizer





cv = CountVectorizer(stop_words = 'english')

words = cv.fit_transform(data['Text'])

sum_words = words.sum(axis=0)





words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]

words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])



color = plt.cm.summer(np.linspace(0, 1, 20))

frequency.tail(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = color)

plt.title("Most Frequently Occuring Words - Top 20")
# cleaning the texts

# importing the libraries for Natural Language Processing



import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
# cleaning the data



corpus = []



for i in range(0, 200):

    review = re.sub('[^a-zA-Z]', ' ', data['Text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)
# vectorizing the data using Tfidf Vectorizer



from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(stop_words='english', max_features = 2000)

X = vectorizer.fit_transform(data['Text'])



# getting the shape of X

print("Shape of X :", X.shape)
true_k = 2

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)

model.fit(X)
print("Top terms per cluster:")



order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()



for i in range(true_k):

    print("Cluster %d:" % i),

    for ind in order_centroids[i, :10]:

        print(' %s' % terms[ind]),

    print



print("\n")

print("Prediction")



# prediction on the Text no. 1

Y = vectorizer.transform(["Claxton hunting first major medal British hurdler Sarah Claxton is confident she can win her first major medal at next month's European Indoor Championships in Madrid.The 25-year-old has already smashed the British record over 60m hurdles twice this season, setting a new mark of 7.96 seconds to win the AAAs title. I am quite confident, said Claxton. But I take each race as it comes. As long as I keep up my training but not do too much I think there is a chance of a medal.Claxton has won the national 60m hurdles title for the past three years but has struggled to translate her domestic success to the international stage. Now, the Scotland-born athlete owns the equal fifth-fastest time in the world this year. And at last week's Birmingham Grand Prix, Claxton left European medal favourite Russian Irina Shevchenko trailing in sixth spot.For the first time, Claxton has only been preparing for a campaign over the hurdles - which could explain her leap in form. In previous seasons, the 25-year-old also contested the long jump but since moving from Colchester to London she has re-focused her attentions. Claxton will see if her new training regime pays dividends at the European Indoors which take place on 5-6 March"])

prediction = model.predict(Y)

print("Cluster number :", prediction)



# Prediction on the Text no.2

Y = vectorizer.transform(["O Sullivan could run in Worlds Sonia O'Sullivan has indicated that she would like to participate in next month's World Cross Country Championships in St Etienne.Athletics Ireland have hinted that the 35-year-old Cobh runner may be included in the official line-up for the event in France on 19-20 March. Provincial teams were selected after last Saturday's Nationals in Santry and will be officially announced this week. O'Sullivan is at present preparing for the London marathon on 17 April. The participation of O'Sullivan, currentily training at her base in Australia, would boost the Ireland team who won the bronze three years agio. The first three at Santry last Saturday, Jolene Byrne, Maria McCambridge and Fionnualla Britton, are automatic selections and will most likely form part of the long-course team. OSullivan will also take part in the Bupa Great Ireland Run on 9 April in Dublin."])

prediction = model.predict(Y)

print("Cluster number :", prediction)