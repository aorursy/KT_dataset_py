# Pandas for file reading/ visualize data

import pandas as pd

import seaborn as sns

import numpy as np

from PIL import Image

#Matplot to visualize data, also Seaborn and pandas do this

import matplotlib.pyplot as plt

# Inline to show images in jupyter notebook

%matplotlib inline
allData = pd.read_csv('../input/Shakespeare_data.csv', sep=',')
wordcld = pd.Series(allData['PlayerLine'].tolist()).astype(str)

mask = np.array(Image.open("../input/william-shakespeare-black-silhouette.jpg"))

# Most frequent words in the data set. Just because. Using a beautiful wordcloud

from wordcloud import WordCloud 

cloud = WordCloud(mask=mask, margin=1,max_font_size=125).generate(' '.join(wordcld.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

plt.show()
allData['Player'].replace(np.nan, 'Other',inplace = True)

allData.head(10)
play_data = allData.groupby('Play').count().sort_values(by='PlayerLine',ascending=False)['PlayerLine']

play_data = play_data.to_frame()

play_data['Play'] = play_data.index.tolist()

play_data.index = np.arange(0,len(play_data)) #changing the index from plays to numbers

play_data.columns =['Lines','Play']
numberPlayers = allData.groupby(['Play'])['Player'].nunique().sort_values(ascending= False).to_frame()

numberPlayers['Play'] = numberPlayers.index.tolist()

numberPlayers.columns = ['NumPlayers','Play']

numberPlayers.index= np.arange(0,len(numberPlayers))

numberPlayers



plt.figure(figsize=(10,10))

ax = sns.barplot(x='NumPlayers',y='Play',data=numberPlayers)

ax.set(xlabel='Number of Players', ylabel='Play Name')

plt.show()
from textblob import TextBlob

data = pd.read_csv('../input/Shakespeare_data.csv')
# Create a new Dataframe and create others off of it

sentiment = pd.DataFrame(columns=('PlayerLinenumber', 'Player', 'PlayerLine'))

sentiment['PlayerLinenumber'] = data['PlayerLinenumber']

sentiment['Player'] = data['Player']

sentiment['PlayerLine'] = data['PlayerLine']

# Drop nulls

sentiment.dropna()

sent2 = sentiment.groupby(['Player'])['PlayerLine'].apply(list)

sent2 = pd.DataFrame(sent2)

# Get the text sentiment of each character

pol = []

for i in sent2['PlayerLine']:

    txt = TextBlob(str.join(' ', i))

    pol.append( (txt.sentiment.polarity)*10 )

# Temporary dataframe then actual dataframe

character_polarity = pd.DataFrame(sent2.index)

character_polarity = pd.DataFrame({'Player':sent2.index, 'Sentiment':pol})
chars_by_polarity = character_polarity.sort_values(by='Sentiment',ascending=False)
sns.set()

f, axes = plt.subplots(figsize=(10, 7))

# Plot a historgram and kernel density estimate

sns.distplot(character_polarity['Sentiment'], color="m")

plt.title('Frequency of Sentiment')

plt.tight_layout()
# Plot the top sentiment

f, ax = plt.subplots(figsize=(6, 15))

sns.set_color_codes("pastel")

sns.barplot(x="Sentiment", y="Player", data=chars_by_polarity.head(25),

            label="Sentiment", color="b")

ax.set(ylabel="Player Name",

       xlabel="Sentiment")

plt.show()
hamlet = character_polarity[character_polarity.Player == 'HAMLET']
print('Hamlets sentiment: ')

hamlet
f, ax = plt.subplots(figsize=(6, 15))

sns.set_color_codes("pastel")

sns.barplot(x="Sentiment", y="Player", data=chars_by_polarity.tail(25),

            label="Sentiment", color="b")

ax.set(ylabel="Player Name",

       xlabel="Sentiment")

plt.show()