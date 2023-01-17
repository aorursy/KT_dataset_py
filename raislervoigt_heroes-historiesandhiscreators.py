import pandas as pd
import numpy as np
df = pd.read_csv('../input/superheroes-nlp-dataset/superheroes_nlp_dataset.csv')
df.head(3)
df['powers_text'].fillna('ni', inplace = True)
df['real_name'].fillna('ni', inplace = True)
df['name'].fillna('ni', inplace = True)
df['name'].isnull().sum()
cdf = df
cdf['history'] = df['history_text'] + ' Name: ' + df['name'] + ' Real Name: ' + df['real_name'] + ' Powers: ' + df['powers_text']

cdf['history'][1]
cdf = cdf[['creator', 'history']]
cdf
cdf.isnull().sum()
cdf = cdf.dropna()
cdf.isnull().sum()
!pip install wordcloud
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip -q
import matplotlib.pyplot as plt
%matplotlib inline 
from wordcloud import WordCloud
text = cdf['history'].sum()
wordcloud = WordCloud(max_font_size=300,background_color="black").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
from pandas_profiling import ProfileReport
profile = ProfileReport(cdf,title = 'Creators Train Dataset report', explorative = True)
profile.set_variable("correlations", None)
profile
# Stopwords

stopwords = [] 
f = open('../input/stopwords-terrier/terrier.txt')

for line in f:
   stopwords.append(line)
stopwords = [x[:-1] for x in stopwords]
print(stopwords)
stopwords = stopwords.append('ni')
# Go to vectors!
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words = stopwords)

features = cdf['history'].to_numpy()
target = cdf['creator'].to_numpy()
train = vectorizer.fit_transform(features)
train
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train, target)
to_pred = ['A boy lost your parents with eight years old, he was train and later became a vigilant in his city!',
           
          '''A baby falls down in a farm, he was born on the planet Krypton and was given the name Kal-El at birth. 
          As a baby, his parents sent him to Earth in a small spaceship
          moments before his planet was destroyed in a natural cataclysm.
          Now he resides in the fictional American city of Metropolis, where he works as a journalist for the Daily Planet.''',
           
        '''A wealthy American business magnate, playboy, and ingenious scientist, 
        he suffers a severe chest injury during a kidnapping. When his captors attempt to force him to build a weapon of mass destruction, 
        he instead creates a mechanized suit of armor to save his life and escape captivity.''',
           
         'The history is simple, He is the Wakanda King'
          ]

# Batman (DC Comics) # Superman (DC Comics) # IronMan (Marvel Comics) # Black Panther (Marvel Comics) 
test = vectorizer.transform(to_pred)
print(clf.predict(test))
