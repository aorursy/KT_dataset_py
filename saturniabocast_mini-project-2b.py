# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
daniel = pd.read_csv("/kaggle/input/daniel-radcliffe/Daniel Radcliffe - Sentiment Analysis Document (5).csv")
papa = pd.read_csv("/kaggle/input/papa-mochi/Papa Mochi - Sentiment Analysis Document (19).csv")
black = pd.read_csv("/kaggle/input/blacklivesmatter/BlackLivesMatter - Sentiment Analysis Document (18).csv")
blake = pd.read_csv("/kaggle/input/blakelivesmatter/BlakeLivesMatter - Sentiment Analysis Document (27).csv")
shut = pd.read_csv("/kaggle/input/shutdownstem/ShutDownSTEM - Sentiment Analysis Document (25).csv")
b = pd.read_csv("/kaggle/input/b-simone/B Simone - Sentiment Analysis Document (45).csv")
stephen = pd.read_csv("/kaggle/input/stephen-miller/Stephen Miller - Sentiment Analysis Document (19).csv")
vita = pd.read_csv("/kaggle/input/vita-trending/Vita - Sentiment Analysis Document (21).csv")
chaz = pd.read_csv("/kaggle/input/capitol-hill-autonomous-zone/Capitol Hill Autonomous Zone - Sentiment Analysis Document (22).csv")
ratm = pd.read_csv("/kaggle/input/rage-against-the-machine/Rage Against the Machine - Sentiment Analysis Document (30).csv")
daniel
papa
black
blake
shut
b
stephen
vita
chaz
ratm
daniel = daniel.drop(['Agreement','Subjectivity','Irony'], axis=1)
daniel
papa = papa.drop(['Agreement','Subjectivity','Irony'], axis=1)
black = black.drop(['Agreement','Subjectivity','Irony'], axis=1)
blake = blake.drop(['Agreement','Subjectivity','Irony'], axis=1)
shut = shut.drop(['Agreement','Subjectivity','Irony'], axis=1)
b = b.drop(['Agreement','Subjectivity','Irony'], axis=1)
stephen = stephen.drop(['Agreement','Subjectivity','Irony'], axis=1)
vita = vita.drop(['Agreement','Subjectivity','Irony'], axis=1)
chaz = chaz.drop(['Agreement','Subjectivity','Irony'], axis=1)
ratm = ratm.drop(['Agreement','Subjectivity','Irony'], axis=1)

dancount = daniel['Polarity'].value_counts()
print(dancount)
papacount = papa['Polarity'].value_counts()
blackcount = black['Polarity'].value_counts()
blakecount = blake['Polarity'].value_counts()
shutcount = shut['Polarity'].value_counts()
bcount = b['Polarity'].value_counts()
stephcount = stephen['Polarity'].value_counts()
vitacount = vita['Polarity'].value_counts()
chazcount = chaz['Polarity'].value_counts()
ratmcount = ratm['Polarity'].value_counts()

print(papacount)
print(blackcount)
print(blakecount)
print(shutcount)
print(bcount)
print(stephcount)
print(vitacount)
print(chazcount)
print(ratmcount)
ts = pd.read_csv("/kaggle/input/trend-sents/trendsents.csv")
ts = ts.drop([10], axis=0)
ts
ts = pd.read_csv("/kaggle/input/trend-sents/trendsents.csv")
ts = ts.drop([10], axis=0)
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
ts.set_index("Topic",inplace=True)
ts.plot.barh(stacked=True, figsize=(10, 6)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Sentiment of Trending Topic Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Topic')
plt.show()
ts2 = pd.read_csv("/kaggle/input/trend-sents2/trendsents2.csv")
ts2
ts2 = ts2.drop(['Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10','Unnamed: 11','Unnamed: 12','Unnamed: 13'], axis=1)
ts2
plt.style.use('seaborn-ticks')
ts2.set_index("Topic",inplace=True)
ts2.plot.barh(stacked=True, figsize=(10, 6)).legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Sentiment of Trending Topic Tweets')
plt.xlabel('Sentiment (Percent of total)')
plt.ylabel('Topic')
plt.show()
