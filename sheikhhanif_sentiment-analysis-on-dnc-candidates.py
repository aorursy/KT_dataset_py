import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS

from textblob import TextBlob

import seaborn as sns
filename='/kaggle/input/dnc-candidates-tweets/sentdata.csv'

data=pd.read_csv(filename,encoding='ISO-8859-1')

data.head()
# checking for null values

data.isnull().sum()
def plot_count():

  """

  This function will retun a bar plot for each candidates tweets count appear in the dataset

  """

  z = {'Tom Steyer': 'Steyer', 'Pete Buttigieg': 'Buttigieg', 'Joe Biden': 'Biden',

       'Elizabeth Warren': 'Warren', 'Bernie Sanders': 'Bernie', 'Amy Klobuchar': 'Amy'}

  

  # mapping authro short name to full name

  x = data.candidate.map(z).unique()

  

  # getting count for each author text appear in train dataset

  y = data.candidate.value_counts().values

  

  plt.bar(x,y, edgecolor='yellow')

  plt.show()

  

  

plot_count()
def plot_words_count():

  """

  This function will return a bar plot for first 30 words counts appears in dataset

  """

  # getting all words and their count of occurances

  words = data.tweet.str.split(expand=True).unstack().value_counts()

  

  # selecting 30 words

  x = words.index.values[2:32]

  y = words.values[2:32]

  

  # plotting barplot

  fig = plt.figure()

  fig.set_figwidth(10)

  plt.bar(x,y, edgecolor = 'yellow')

  

plot_words_count()
def word_cloud_viz():

  """

  This function will return word_cloud visualization of words for each candidate tweet

  """

  # python list that store text of three author

  ts = data[data.candidate=="Tom Steyer"]["tweet"].values

  pb = data[data.candidate=="Pete Buttigieg"]["tweet"].values

  jb = data[data.candidate=="Joe Biden"]["tweet"].values



  ew = data[data.candidate=="Elizabeth Warren"]["tweet"].values

  bs = data[data.candidate=="Bernie Sanders"]["tweet"].values

  ak = data[data.candidate=="Amy Klobuchar"]["tweet"].values

  # z = {'EAP': 'Edgar Allen Poe', 'MWS': 'Mary Shelley', 'HPL': 'HP Lovecraft'}



  # plotting Top Steyer word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(ts))

  plt.title("Tom Steyer", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')



  

  # plotting Pete Buttigieg word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(pb))

  plt.title("Pete Buttigieg", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')

  

  

  # plotting Joe Biden word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(jb))

  plt.title("Joe Biden", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')





  # plotting Elizabeth Warren word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(ew))

  plt.title("Elizabeth Warren", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')





  

  # plotting Bernie Sanders word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(bs))

  plt.title("Bernie Sanders", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')

  

  

  # plotting Amy Klobuchar word_cloud

  plt.figure(figsize=(15, 10))

  wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size=40)

  wc.generate(" ".join(ak))

  plt.title("Amy Klobuchar", fontsize=20)

  plt.imshow(wc.recolor( colormap= 'Pastel2' , random_state=17), alpha=0.98)

  plt.axis('off')

  

  

word_cloud_viz()
def sentiment(df_tweet_polarity_desc):

    if df_tweet_polarity_desc['sentiment'] > 0:

        val = "Positive"

    elif df_tweet_polarity_desc['sentiment'] == 0:

        val = "Neutral"

    else:

        val = "Negative"

    return val





def sentiment_analysis(candidate_name):

  tweets= data.loc[(data['candidate']== candidate_name), ['tweet']]

  bloblist_desc = list()

  df_tweet_str=tweets['tweet'].astype(str)

  for row in df_tweet_str:

      blob = TextBlob(row)

      bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))

      df_tweet_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])



  df_tweet_polarity_desc['Sentiment_Type'] = df_tweet_polarity_desc.apply(sentiment, axis=1)



  plt.figure(figsize=(10,10))

  plt.title(candidate_name)

  sns.set_style("whitegrid")

  ax = sns.countplot(x="Sentiment_Type", data=df_tweet_polarity_desc)
candidate_names = ["Tom Steyer", "Pete Buttigieg", "Joe Biden", "Elizabeth Warren", "Bernie Sanders", "Amy Klobuchar"]

for cn in candidate_names:

  sentiment_analysis(cn)