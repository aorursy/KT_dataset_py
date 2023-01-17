import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("/kaggle/input/covid19-tweets/covid19_tweets.csv").iloc[0:10000]

import os

! apt-get update -qq > /dev/null  



# Install java

! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

! java -version

 # TODO make install SILENT! and upgrade

! pip install  nlu==2.5rc1 -qq > /dev/null  



import nlu
import nlu



emotion_predictions = nlu.load('emotion').predict(df)

emotion_predictions['category'].value_counts().plot.bar(title='Predicted emotion label count in dataset')
counts = emotion_predictions.groupby(['source'])['category'].value_counts()

counts[counts > 100].plot.bar(figsize=(20,8), title= 'Emotion tweet counts grouped by tweet device source')
counts = emotion_predictions.groupby(['user_location'])['category'].value_counts()

counts[counts > 25].plot.bar(figsize=(20,6), title= 'Emotion tweet counts grouped by user location')