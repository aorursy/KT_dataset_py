import numpy as np
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/18171739/unicodedecodeerror-when-reading-csv-file-in-pandas-with-python
tweets = pd.read_csv('../input/TweetsNBA.csv', encoding = "ISO-8859-1")
tweets = tweets.loc[tweets['lang'] == 'en']
#https://textblob.readthedocs.io/en/dev/quickstart.html
#https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.en.sentiments.PatternAnalyzer
tweets['polarity'] = tweets['text'].apply(lambda x: TextBlob(x).polarity)
tweets['subjectivity'] = tweets['text'].apply(lambda x: TextBlob(x).subjectivity)
Unverified = tweets.loc[tweets['verified'] == False]
Verified = tweets.loc[tweets['verified'] == True]
print('Verified Subjectivity: {}'.format(Verified['subjectivity'].mean()))
print('Unverified Subjectivity: {}'.format(Unverified['subjectivity'].mean()))
#https://stackoverflow.com/questions/35595710/splitting-timestamp-column-into-seperate-date-and-time-columns
tweets['created_at'] = pd.to_datetime(tweets['created_at'])
#https://stackoverflow.com/questions/16266019/python-pandas-group-datetime-column-into-hour-and-minute-aggregations/32066997
Game3 = tweets.groupby(tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Total Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Game3)
plt.gcf().autofmt_xdate()
plt.show()
tweets['cavs'] = tweets['text'].str.contains('cav',regex = False, case = False)
Cavs_Tweets = tweets.loc[tweets['cavs'] == True]
tweets['warriors'] = tweets['text'].str.contains('warriors',regex = False, case = False)
Warriors_Tweets = tweets.loc[tweets['warriors'] == True]
Game3_Cavs = Cavs_Tweets.groupby(Cavs_Tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Cavs Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Game3_Cavs)
plt.gcf().autofmt_xdate()
plt.show()
Games3_Warriors = Warriors_Tweets.groupby(Warriors_Tweets['created_at'].dt.minute)['polarity'].mean()

plt.figure(figsize = (20,10))
plt.suptitle('Warriors Game 3 Polarity', fontsize = 24)
plt.xlabel('Time', fontsize = 20)
plt.ylabel("Polarity", fontsize = 20)
plt.plot(Games3_Warriors)
plt.gcf().autofmt_xdate()
plt.show()