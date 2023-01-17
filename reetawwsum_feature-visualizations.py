# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/offenders.csv", encoding='latin1')
df.head()
df.columns
df.dtypes
df.shape
features = ["Last Name", "First Name"]



for feature in features:

    wordCloud = WordCloud(

            background_color='white'

        ).generate(" ".join(df[feature]))



    plt.figure()

    plt.imshow(wordCloud)

    plt.title(feature)

    plt.axis("off")
df["Age"].value_counts().sort_index().plot()
date_df = pd.DataFrame()



date_df["Month"], date_df["Date"], date_df["Year"] = df["Date"].str.split("/").str



sns.factorplot("Month", data=date_df, kind="count")



plt.figure()

dateCounts_df = date_df["Date"].value_counts()

dateCounts_df.index = dateCounts_df.index.astype(int)

plt.figure()

dateCounts_df.sort_index().plot()

plt.title("Date")



plt.figure()

date_df["Year"].value_counts().sort_index().plot()

plt.title("Year")
for i in range(0,len(df)):

    race = df["Race"].iloc[i]

    if race == "White ":

        df["Race"].iloc[i] = "White"

    elif race == "Hispanic ":

        df["Race"].iloc[i] = "Hispanic"

    else:

        pass

    

sns.factorplot("Race", data=df, kind="count")
wordCloud = WordCloud(background_color="white").generate(" ".join(df["County"]))



plt.imshow(wordCloud)

plt.axis("off")
wordCloud = WordCloud(

        stopwords=STOPWORDS,

        background_color='white'

    ).generate(" ".join(df["Last Statement"]))



plt.imshow(wordCloud)

plt.axis("off")
lastWords = df["Last Statement"].map(lambda x: x.split()[-1])



wordCloud = WordCloud(

        stopwords=STOPWORDS,

        background_color='white'

    ).generate(" ".join(lastWords[lastWords != "None"]))



plt.imshow(wordCloud)

plt.axis("off")