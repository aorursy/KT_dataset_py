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
# read data

df = pd.read_csv('../input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv')
df.columns
# visualization library

import seaborn as sns

#  pair plot

sns.pairplot(df)
sns.regplot(x=df['lat'], y=df['Reviewer_Score'])
sns.regplot(x=df['lng'], y=df['Reviewer_Score'])
# Reviewe_Score counts

sns.distplot(df["Reviewer_Score"],kde=False,bins=15)
df.shape
df['Reviewer_Score'].min() , df['Reviewer_Score'].max(), df['Reviewer_Score'].mean()
countries = df["Reviewer_Nationality"].value_counts()[df["Reviewer_Nationality"].value_counts() > 100]

g = df.groupby("Reviewer_Nationality").mean()

g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values(ascending=False)[:10].plot(kind="bar",ylim=(8.395076569886239,9),title="Top Reviewing Countries")
g.loc[countries.index.tolist()]["Reviewer_Score"].sort_values()[:10].plot(kind="bar",ylim=(2.5,8.395076569886239),title="least Reviewing Countries")
def country_ident(st):

    last = st.split()[-1]

    if last == "Kingdom": return "United Kingdom"

    else: return last

    

df["Hotel_Country"] = df["Hotel_Address"].apply(country_ident)

df.groupby("Hotel_Country").mean()["Reviewer_Score"].sort_values(ascending=False)
best_hotels = df.groupby('Hotel_Name')['Reviewer_Score'].mean().sort_values(ascending=False).head(10)

best_hotels.plot(kind="bar",color = "Green")
from datetime import datetime

df["Review_Date_Month"] = df["Review_Date"].apply(lambda x: x[5:7])

df[["Review_Date","Reviewer_Score"]].groupby("Review_Date").mean().plot(figsize=(15,5))
from wordcloud import WordCloud

import matplotlib.pyplot as plt

def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color = 'white',

        max_words = 200,

        max_font_size = 40, 

        scale = 3,

        random_state = 42

    ).generate(str(data))



    fig = plt.figure(1, figsize = (20, 20))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize = 20)

        fig.subplots_adjust(top = 2.3)



    plt.imshow(wordcloud)

    plt.show()

    

# print wordcloud

show_wordcloud(df['Positive_Review'])
# most positive

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(analyzer = "word",stop_words = 'english',max_features = 20,ngram_range=(2,2))

most_positive_words = cv.fit_transform(df['Positive_Review'])

temp1_counts = most_positive_words.sum(axis=0)

temp1_words = cv.vocabulary_

temp1_words
show_wordcloud(df['Negative_Review'])
cv = CountVectorizer(analyzer = "word",stop_words = 'english',max_features = 20,ngram_range=(2,2))

most_negative_words = cv.fit_transform(df['Negative_Review'])

temp1_counts = most_negative_words.sum(axis=0)

temp1_words = cv.vocabulary_

temp1_words
# extrating nights from tag

def splitString(string):

    array = string.split(" ', ' ")

    array[0] = array[0][3:]

    array[-1] = array[-1][:-3]

    if not 'trip' in array[0]:

        array.insert(0,None)

    try:

        return float(array[3].split()[1])

    except:

        return None



df["Nights"] = df["Tags"].apply(splitString)

sns.jointplot(data=df,y="Reviewer_Score",x="Nights",kind="reg")
df['Leisure'] = df['Tags'].map(lambda x: 1 if ' Leisure trip ' in x else 0)

df['Business'] = df['Tags'].map(lambda x: 2 if ' Business trip ' in x else 0)

df['Trip_type'] = df['Leisure'] + df['Business']