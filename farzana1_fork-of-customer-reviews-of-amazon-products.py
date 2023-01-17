import seaborn as sns

from textblob import TextBlob

from matplotlib import pyplot as plt

import pandas as pd
df = pd.read_csv("../input/data3.csv")

df.head()
from textblob import TextBlob #processing textual data



bloblist_desc = list()



df_review_str=df['reviews.text'].astype(str)

for row in df_review_str:

    blob = TextBlob(row)

    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))

    df_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['Review','sentiment','polarity'])

 

def f(df_polarity_desc):

    if df_polarity_desc['sentiment'] > 0:

        val = "Positive Review"

    elif df_polarity_desc['sentiment'] == 0:

        val = "Neutral Review"

    else:

        val = "Negative Review"

    return val



df_polarity_desc['Sentiments'] = df_polarity_desc.apply(f, axis=1)



plt.figure(figsize=(8, 8))



sns.set_style("whitegrid")



ax = sns.countplot(x="Sentiments", data=df_polarity_desc)
plt.show()
#df = pd.read_csv("C:\\Users\Microsoft\Documents\data3.csv")

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib

import numpy as np

matplotlib.style.use('ggplot')

plt.figure(figsize=(15,6))

N = 48

x1 = np.random.random(N)

y1= np.random.random(N)

#x1 = np.random.rand(N,M)

#y1 = np.random.rand(N,M)

X = np.arange(N)



plt.bar(X, x1, color ='r', hatch ='x', label="Negative")

plt.bar(X, x1+y1, bottom = x1, color ='c', hatch ='/', label="Positive")



plt.xlabel('Number of Products')

plt.ylabel('Number of Reviews')

plt.title('Amazon Products Sentiment')



plt.legend()
df = pd.read_csv("../input/Percentage.csv")

df