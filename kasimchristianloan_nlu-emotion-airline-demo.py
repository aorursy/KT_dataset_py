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
nlu_emotion_df = nlu.load('emotion').predict(df)

nlu_emotion_df['category'].value_counts().plot.bar(title='Predicted emotion labels count in dataset')

counts = nlu_emotion_df.groupby('user_timezone')['category'].value_counts()

counts[counts >10].plot.bar(figsize=(25,8),title='Emotion tweet counts by user time zone')
nlu_emotion_df.groupby('airline')['category'].value_counts().plot.bar(figsize=(20,8), title='Emotion tweet counts grouped by airline')
