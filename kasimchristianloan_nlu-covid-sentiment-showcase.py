import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv")

df
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

sentiment_predictions = nlu.load('sentiment').predict(df)

sentiment_predictions['sentiment'].value_counts().plot.bar(title='Count of predicted sentiment labels')
counts = sentiment_predictions.groupby('source')['sentiment'].value_counts()

counts[counts>100].plot.bar(figsize=(20,8), title='Sentiment tweet counts grouped by tweet source')
counts = sentiment_predictions.groupby(['user_location'])['sentiment'].value_counts()

counts[counts >1000 ].plot.bar(figsize=(20,6), title='Sentiment tweet counts grouped by user location')