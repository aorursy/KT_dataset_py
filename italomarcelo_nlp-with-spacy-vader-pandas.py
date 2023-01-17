!pip install feedparser

import feedparser
!pip install vaderSentiment

import vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
phrases = [

"I would feel the same in your situation, but we will sort this out",

"I know how frustrating it can be – let’s see how I can help you",

"I completely understand how frustrating it is",

"I appreciate how difficult it is",

"We will work to resolve the problem. You just enjoy your (birthday/holidays/Christmas break, etc.), and I will be in touch shortly.",

"We are keen to resolve this as much as you are",

"If I were in your position, I would feel exactly the same",

"I appreciate you bringing this to our attention…",

"I will contact you as soon as we have had an update.",

"Definitely, you are making perfect sense.",

"I can assure you that this will absolutely happen.",

"We will help you get this issue resolved.",

"Apologies for the wait, I appreciate your patience.",

"Don’t worry, I can see why you did that.",

"Yes, that would certainly frustrate me too.",

"What I would do in this situation i"

"How do you feel about fear",

"I know others who have been in your situation and what we did to successfully help them",

"Just so we know what we’re aiming for, what would be the best-case scenario for you.",

"Absolutely, I can certainly fix that for you.",

"I will quickly put this into action for you and then everything will be back to normal.",

"Just so I can clarify and help you.",

"I’m afraid that we cannot offer you X, but what I can do for you",

"I am going to take care of this for you.",

"That does sound frustrating, let’s see what I can do to help.",    

"I love the team and how they played last night 💘" ,

"What a fine day I am having today :-) :-)",

"I am laughing like crazy lol",

"He was not very good at the play",

"He is kinda bored",

]



readSentiment = SentimentIntensityAnalyzer()
for txt in phrases:

  print(readSentiment.polarity_scores(txt))
def getSentiment(phrase):

  s = readSentiment.polarity_scores(phrase)

  if s['compound'] <= -0.05:

    sentiment = 0

  elif s['compound'] >= 0.05:

    sentiment = 1

  else:

    sentiment = 2

  return sentiment, s
t1 = getSentiment("He is kinda bored")

t1
t1 = getSentiment("The book was good")

t1
t1 = getSentiment("Taco's Tuesday")

t1
sentiments = ['Negative', 'Positive', 'Neutral']

for txt in phrases:

  print(sentiments[getSentiment(txt)[0]], ' - ', txt)
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
titles = []

summaries = []

for feed in feeds:

  for content in feed.entries:

    titles.append(content.title)

    try:

      summaries.append(content.summary)

    except:

      summaries.append(content.title)
for txt in titles:

  print(sentiments[getSentiment(txt)[0]], ' - ', txt)
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

  txt = str(txt).strip()

  if show:

    print('ORIGINAL: ', original)

    print('   TEXT CLEANNED: ', txt)

  return txt
titlesClean = [cleaningText(title) for title in titles]
for txt in titlesClean:

  print(sentiments[getSentiment(txt)[0]], ' - ', txt)
import pandas as pd
sent = []

for txt in titles:

  sent.append(getSentiment(txt)[0])



dfT = pd.DataFrame()

dfT['title'] = titles

dfT['titleClean'] = titlesClean

dfT['sentimentTitle'] = sent
dfT.head()
dfTitle = dfT[['sentimentTitle','title']].groupby('sentimentTitle').count()

dfTitle
import plotly.express as px
dfTitle.values.reshape(-1)
colors=['orange', 'Darkblue', 'Darkred']

px.pie(names=sentiments, values=dfTitle.values.reshape(-1), title='Sentiment Analysis - Titles', 

       color_discrete_sequence=colors)
summariesClean = [cleaningText(summary) for summary in summaries]
sent = []

for txt in summaries:

  sent.append(getSentiment(txt)[0])



dfS = pd.DataFrame()

dfS['summary'] = summaries

dfS['summaryClean'] = summariesClean

dfS['sentimentSummary'] = sent
dfSummary = dfS[['sentimentSummary','summary']].groupby('sentimentSummary').count()

dfSummary
colors=['Darkblue', 'orange', 'Darkred']

px.pie(names=sentiments, values=dfSummary.values.reshape(-1), title='Sentiment Analysis - Summaries', 

       color_discrete_sequence=colors)
def tokenizeStr(original):

  txt2 = nlp(original) # creating a word's list

  txt2 = [token.lemma_ for token in txt2 if not nlp.vocab[token.text].is_stop]

  punct = string.punctuation

  stopwords = list(STOP_WORDS)

  ws = string.whitespace

  txt2 = [word for word in txt2 if word not in stopwords and word not in punct if len(word)>2]

  return txt2
# Example tokenizing sentence and 

tokenizeStr('Home sweet Home'), tokenizeStr("Choose a study category to start")
[a*b for b in range(1,3) for a in range(4,6)]
wordsT = [word for i in range(0, len(dfT)-1) for word in tokenizeStr(dfT.iloc[i].titleClean) if str(word).strip() != '']

wordsTn = pd.value_counts(wordsT)
wordsS = [word for i in range(0, len(dfS)-1) for word in tokenizeStr(dfS.iloc[i].summaryClean) if str(word).strip() != '']

wordsSn = pd.value_counts(wordsS)
wordlist = pd.value_counts(wordsT+wordsS)
topW = pd.DataFrame(data={'tag': wordlist.index, 'count':wordlist.values})
topW[:10]
px.bar(topW[:10], y='tag', x='count', orientation='h', 

       title='Top 10 words', color='tag')
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfT.title)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from titles')

  

plt.show() 
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfS['summaryClean'])) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from summaries')

  

plt.show() 
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfT[dfT.sentimentTitle == 1].title)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from titles - sentiment Positive')

  

plt.show() 
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfT[dfT.sentimentTitle == 0].title)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from titles - sentiment negative')

  

plt.show()
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfS[dfS.sentimentSummary == 1].summaryClean)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from summaries - sentiment positive')

  

plt.show()
wordcloud = WordCloud(width = 800, height = 600, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(' '.join(dfS[dfS.sentimentSummary == 0].summaryClean)) 

  

# plot the WordCloud image                        

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.title('words from summaries - sentiment negative')

  

plt.show()