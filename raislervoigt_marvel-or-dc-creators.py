



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/superheroes-nlp-dataset/superheroes_nlp_dataset.csv')

df.head(2)
cdf = df[['creator', 'history_text']]

cdf.head(3)
print('Null Values')

print(cdf.isnull().sum())

print('________________________')

print(cdf['creator'].value_counts())
cdf = cdf.dropna()

print(cdf.isnull().sum())

print('--------------------------------')

print(cdf['creator'].value_counts())
mask1 = cdf.loc[(cdf['creator'] == 'Marvel Comics' )]

mask2 = cdf.loc[(cdf['creator'] == 'DC Comics' )]

frames = [mask1, mask2]

cdf = pd.concat(frames)

cdf
print(cdf.isnull().sum())

print('--------------------------------')

print(cdf['creator'].value_counts())
features = cdf['history_text'].to_numpy()

target = cdf['creator'].to_numpy()
Cloud_Marvel = cdf['history_text'].loc[(cdf.creator == 'Marvel Comics')]

Cloud_Marvel = Cloud_Marvel.sum()

Cloud_DC = cdf['history_text'].loc[(cdf.creator == 'DC Comics')]

Cloud_DC = Cloud_DC.sum()
import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=300,background_color="black").generate(Cloud_Marvel)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
import matplotlib.pyplot as plt

%matplotlib inline 

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=300,background_color="black").generate(Cloud_DC)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
stopwords = [] 

f = open('../input/stopwords-terrier/terrier.txt')



for line in f:

   stopwords.append(line)
# Based on our cloud of words

stopwords.append('men')

stopwords.append('help')
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words = stopwords);

features = vectorizer.fit_transform(features);
from sklearn.model_selection import train_test_split





X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.10, random_state=42)
from sklearn.naive_bayes import MultinomialNB

Naive_Bayes = MultinomialNB()

Naive_Bayes.fit(X_train, y_train)
X_train_dense = X_train.todense()

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train_dense, y_train)
from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression( solver = 'liblinear', penalty='l1')

LR_model.fit(X_train, y_train)
X_test_dense = X_test.todense()



print('Naive Bayes MultinomialNB:',Naive_Bayes.score(X_test, y_test))

print('Naive Bayes GaussianNB:',gnb.score(X_test_dense, y_test))

print('Logistic Regression solver = liblinear:', LR_model.score(X_test, y_test))

DC = ['A boy lost your parents with eight years old, he was train and later became a vigilant in his city!',

           

          '''A baby falls down in a farm, he was born on the planet Krypton and was given the name Kal-El at birth. 

          As a baby, his parents sent him to Earth in a small spaceship

          moments before his planet was destroyed in a natural cataclysm.

          Now he resides in the fictional American city of Metropolis, where he works as a journalist for the Daily Planet.''',



           'He is a vigilant in Star City, His main weapon is a bow and arrow, his favorite color is Green',

           

           'He is the fastest man alive, sometimes actually, when he was a kid, his mom was murdered from a yellow man, a yellow thing',          

          ]



Marvel = ['''A wealthy American business magnate, playboy, and ingenious scientist, 

        he suffers a severe chest injury during a kidnapping. When his captors attempt to force him to build a weapon of mass destruction, 

        he instead creates a mechanized suit of armor to save his life and escape captivity.''',

           

         'The history is simple, He is the Wakanda King',

      

      'His Father is Odin',

           

'He was a normal scientist falling in love for a beautiful scientist, but now when he is Angry, he become a green monster, and everybody call him HULK',

          

          'He is a good person, a good hero, an old hero, he is a captain, a leader, currently he is an avenger.'

     ]

Marvel = vectorizer.transform(Marvel)

DC = vectorizer.transform(DC)

DC_dense = DC.todense()

Marvel_dense = Marvel.todense()

print('Marvel Predictions')

print('')

print('MultinomialNB: ', Naive_Bayes.predict(Marvel))

print('')

print('Logistic Regression: ', LR_model.predict(Marvel))

print('')

print('GaussianNB: ', gnb.predict(Marvel_dense))

print('')

print('DC Predictions')

print('')

print('MultinomialNB: ',Naive_Bayes.predict(DC))

print('')

print('Logistic Regression: ', LR_model.predict(DC))

print('')

print('GaussianNB: ', gnb.predict(DC_dense))