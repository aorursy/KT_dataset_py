import os

! apt-get update -qq > /dev/null  



# Install java

! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

! java -version

! pip install  nlu==2.5rc1 -qq > /dev/null   



import nlu 
import nlu

import pandas as pd

df = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')

df
sentiment_predictions = nlu.load('sentiment').predict(df)

sentiment_predictions['sentiment'].value_counts().plot.bar(title='Count of each sentiment label predicted')
sentiment_predictions.groupby('airline')['sentiment'].value_counts().plot.bar(figsize=(20,8), title = 'Sentiment counts grouped by tweet airline')
counts  = sentiment_predictions.groupby('tweet_location')['sentiment'].value_counts()



counts[counts>50].plot.bar(figsize=(20,8), title = 'Sentiment counts grouped by tweet location')
counts = sentiment_predictions.groupby('user_timezone')['sentiment'].value_counts()

counts[counts>100].plot.bar(figsize=(20,8), title='Sentiment counts grouped by user location')