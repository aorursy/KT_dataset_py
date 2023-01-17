# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("/kaggle/input/nba-injuries-2010-2018/injuries.csv")

data.head()
achil=data[(data["Notes"].str.find("Achil")!=-1)]

achil
achil.groupby("Relinquised").count().sort_values(by="Date",ascending=False)["Date"]
tornachilles=achil[(achil["Notes"].str.find("torn")!=-1)]

print(len(tornachilles))
tornachilles
nojeff=tornachilles[tornachilles["Relinquised"]!="Jeffery Taylor / Jeff Taylor (b)"]

nojeff[nojeff["Notes"].str.find("sur")==-1]
ankles=data[(data["Notes"].str.find("ankle")!=-1)|(data["Notes"].str.find("Ankle")!=-1) ]

ankles.shape[0]
anklesurgery=ankles[ankles["Notes"].str.find("sur")!=-1]

anklesurgery.shape[0]
anklesurgery
howmanyinjuries=data.groupby("Relinquised").count().sort_values(by="Date",ascending=False)["Date"]

howmanyinjuries[0:30]
knee=data[(data["Notes"].str.find("knee")!=-1)|(data["Notes"].str.find("Knee")!=-1) ]

knee.shape[0]
knee
howmanyknee=knee.groupby("Relinquised").count().sort_values(by="Date",ascending=False)["Date"]

howmanyknee[0:30]
data[data["Relinquised"]=="Lebron James"]
data[data["Relinquised"]=="Paul Pierce"]
texts=""

for text in data["Notes"]:

    texts=texts+text+" "


from wordcloud import STOPWORDS

STOPWORDS.add("one")

STOPWORDS.add("want")

STOPWORDS.add("didn")

STOPWORDS.add("lot")

STOPWORDS.add("don")

STOPWORDS.add("think")

STOPWORDS.add("anything")

STOPWORDS.add("someone")

STOPWORDS.add("know")

STOPWORDS.add("nan")

STOPWORDS.add("will")

STOPWORDS.add("well")

STOPWORDS.add("much")

STOPWORDS.add("say")

STOPWORDS.add("nan")

STOPWORDS.add("us")

from wordcloud import WordCloud

wordcloud = WordCloud(max_font_size=40,stopwords=STOPWORDS).generate(texts.lower())

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,40))

plt.imshow(wordcloud)
STOPWORDS.add("returned")

STOPWORDS.add("sprained")

STOPWORDS.add("left")

STOPWORDS.add("right")

STOPWORDS.add("sore")

STOPWORDS.add("lineup")

STOPWORDS.add("dnp")

STOPWORDS.add("injury")

wordcloud = WordCloud(max_font_size=40,stopwords=STOPWORDS).generate(texts.lower())

plt.figure(figsize=(20,40))

plt.imshow(wordcloud)