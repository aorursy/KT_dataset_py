# !pip install -U spacy

!pip install feedparser

import feedparser
sources = [

          "http://rss.cnn.com/rss/edition_travel.rss",

          "http://rss.cnn.com/rss/edition_world.rss",

          "http://rss.cnn.com/rss/money_news_international.rss",

          "http://rss.cnn.com/rss/edition_technology.rss",

          "http://rss.cnn.com/rss/edition_entertainment.rss"

          'https://www.espn.com/espn/rss/nfl/news',

          'https://www.espn.com/espn/rss/nba/news',

          'https://www.espn.com/espn/rss/rpm/news',

          'https://www.espn.com/espn/rss/soccer/news',

          'https://www.espn.com/espn/rss/mlb/news',

          'https://www.espn.com/espn/rss/nhl/news',

          'https://www.espn.com/espn/rss/poker/master',

           'http://rss.cnn.com/rss/edition_sport.rss',

           'http://rss.cnn.com/rss/edition_football.rss',

           'http://rss.cnn.com/rss/cnn_latest.rss',

           'http://rss.cnn.com/rss/edition_space.rss',

           'http://rss.cnn.com/rss/edition.rss',

           'http://rss.cnn.com/rss/edition_africa.rss',

           'http://rss.cnn.com/rss/edition_americas.rss'



          

]
feeds = []

for s in sources:

  feed = feedparser.parse(s)

  feeds.append(feed)

  print('Cols: ', feed.entries[1].keys())
titles = []

for feed in feeds:

  for content in feed.entries:

    titles.append(content.title)
print(*titles[:3], sep='\n')
import spacy

from  spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
import re

import string



def cleaningText(original, show=False):

  txt = original

  txt = txt.lower() # lowercase

  txt = re.sub('@','',txt) # remove @ 

  txt = re.sub('\[.*\]','',txt) # remove contents between brackets

  txt = re.sub('<.*?>+','',txt) # remove contents between less and more signs

  txt = re.sub('https?://\S+|www\.\S+', '', txt) # remove URLs

  txt = re.sub(re.escape(string.punctuation), '', txt) # remove punctuation

  txt = re.sub(r'[^a-zA-Z ]+', '', txt) # remove numbers

  txt = re.sub('\n', '', txt) # remove line break

  txt2 = nlp(txt) # creating a word's list

  txt2 = [token.lemma_ for token in txt2 if not nlp.vocab[token.text].is_stop]

  punct = string.punctuation

  stopwords = list(STOP_WORDS)

  txt2 = [word for word in txt2 if word not in stopwords and word not in punct]



  if show:

    print('ORIGINAL: ', original)

    print('   TEXT CLEANNED: ', txt)

    print('   WORD LIST: ', txt2[:20])

  return txt,txt2
titlesClean = [cleaningText(t) for t in titles]
for title, __ in titlesClean[:5]:

  if title:

    print('-', title)

    for pos in nlp(title):

      print(f'==>{pos.text} ({pos.pos_}) ', end='')

    print()
for title, __ in titlesClean[:5]:

  if title:

    print('-', title)

    for pos in nlp(title):

      if pos.pos_ == 'VERB':

        print(f'==>{pos.text} ({pos.pos_}) ', end='')

    print()
for title, __ in titlesClean[:10]:

  if title:

    print('-', title)

    for e in nlp(title).ents:

      print(f'==>{e}: {e.label_} ({e.label}) ', end='')

    print()
entityList = []



for title, __ in titlesClean[:15]:

  if title:

    print('-', title)

    for e in nlp(title).ents:

      if e.label_ == 'ORG':

        entityList.append(e)

        print(f'==>{e} ({e.label_}) ', end='')

    print('')       
import pandas as pd

for e in pd.Series(entityList).unique():

  s1 = nlp(str(e))

  for v in pd.Series(entityList).unique():

    s2 = nlp(str(v))

    try:

      print(f'Similarity between {e} and {v}: {s2.similarity(s1)}')

    except:

      pass
from spacy import displacy

from IPython.display import SVG, display

def showSVG(s):

  display(SVG(x))
for title, __ in titlesClean[:15]:

  x= displacy.render(nlp(title), style = "ent")

  showSVG(x)
doc = nlp("giant robot comes to life in japan")

for chunk in doc.noun_chunks:

    print(chunk.text, chunk.root.text, chunk.root.dep_,

            chunk.root.head.text)
doc = nlp("giant robot comes to life in japan")

x = displacy.render(doc)

showSVG(x)
from textblob import TextBlob



# setup 

colors = ['Green', 'Goldenrod']

explode = (0.01, 0.01)

labels = ['Positive', 'Negative']



pos = 0

neg = 0

listWL = []

# to DataFrame in Machine Learning

phrase = []

status = []



for title, wlist in titlesClean:

  # creating wordlist

  for word in wlist:

    if len(word.strip()):

      listWL.append(word)

  t = TextBlob(title)

  polarity = t.sentiment.polarity

  if polarity != 0:

    phrase.append(title) 

    if polarity > 0:

        pos += 1

        status.append(1)

    else:

        neg += 1

        status.append(-1)



# DataFrame to predicts Machine Learning

df = pd.DataFrame()

df['text'] = phrase

df['sentiment'] = status

    
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(15,8))

plt.pie([pos, neg], labels=labels, colors=colors, startangle=90, explode = explode, autopct = '%1.2f%%')

plt.axis('equal') 

plt.title('RSS CNN')

plt.show()
fig, ax1 = plt.subplots(sharey=True, figsize=(15,9))

sns.barplot(x=pd.Series(listWL).value_counts()[:20].index, 

            y=pd.Series(listWL).value_counts()[:20].values,

            ax=ax1).set_title('Word List')

plt.xlabel('word')

plt.ylabel('count')

plt.xticks(rotation=80)
# sklearn

# ML classifiers

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC





# ML selecao de dados de treino e teste

from sklearn.model_selection import train_test_split, cross_val_score

# confusion matrics

from sklearn.metrics import confusion_matrix

# metrics

from sklearn import metrics

# vetorizador

from sklearn.feature_extraction.text import TfidfVectorizer
df.info()
df.head()
df.sample(3)
df.isnull().sum()
df.sentiment.plot()

X = df.text

y = df.sentiment



Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)



txtvector = TfidfVectorizer()

vXtrain = txtvector.fit_transform(Xtrain)

vXtest = txtvector.transform(Xtest)
# training data

classifiers = [

               SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42), 

               LinearSVC(),

               MultinomialNB(),

               RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),

               LogisticRegression(random_state=0),

               

]

models = pd.DataFrame(columns=['Name', 'Score'])

for classifier in classifiers:

  model = classifier

  model.fit(vXtrain, ytrain)

  pred = model.predict(vXtest)

  models = models.append({'Name': str(model).split('(')[0], 'Score': model.score(vXtrain, ytrain)}, ignore_index=True)

  print(model)

  print(metrics.classification_report(ytest.values, pred, target_names=['Negative', 'Positive'], zero_division=0))
sns.boxplot(x='Name', y='Score', data=models)

chart = sns.stripplot(x='Name', y='Score', data=models, 

              size=8, jitter=True, edgecolor="gray", linewidth=2)

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)