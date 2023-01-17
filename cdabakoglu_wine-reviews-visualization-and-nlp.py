# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly import tools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/winemag-data-130k-v2.csv")
df.info()
df.head() # First 5 rows of our dataset
# Let's delete "Unnamed: 0" column
df.drop(["Unnamed: 0"], axis=1, inplace=True)
plt.figure(figsize=(16,7))
sns.set(style="darkgrid")
sns.barplot(x=df.country.value_counts()[:10].index, y=df.country.value_counts()[:10].values)
plt.xlabel("Countries")
plt.ylabel("Number of Wine")
plt.show()
plt.figure(figsize=(16,7))
g = sns.barplot(x=df.groupby("country").mean().sort_values(by="points",ascending=False).points.index[:10], y=df.groupby("country").mean().sort_values(by="points",ascending=False).points.values[:10], palette="gist_ncar")
plt.xlabel("Countries")
plt.ylabel("Average Points")
plt.title("Average Points Top 10")
ax=g
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')
plt.show()
plt.figure(figsize=(16,7))
g = sns.barplot(x=df.groupby("country").mean().sort_values(by="price",ascending=False).price.index[:10], y=df.groupby("country").mean().sort_values(by="price",ascending=False).price.values[:10], palette="Blues_r")
plt.xlabel("Countries")
plt.ylabel("Average Price (US Dollar)")
plt.title("Average Price Top 10")
ax=g
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')
plt.show()
df2 = df[np.isfinite(df["price"])]
df2["points/price"] = df2.points / df2.price
df2.groupby("country").mean().sort_values(by="points/price", ascending=False)

plt.figure(figsize=(16,7))
g = sns.barplot(x=df2.groupby("country").mean().sort_values(by="points/price", ascending=False)["points/price"].index[:10], y=df2.groupby("country").mean().sort_values(by="points/price", ascending=False)["points/price"].values[:10], palette="jet_r")
plt.xlabel("Countries")
plt.xticks(rotation= 45, ha="right")
plt.ylabel("Points / Price Ratio")
ax=g
for p in ax.patches:
    ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')
plt.show()
plt.figure(figsize=(12,6))
sns.boxplot(x=df.points)
plt.title("Points Boxplot")
plt.show()
top20Points = df.sort_values(by="points", ascending=False).head(20)

for i in range(20):
    print("{} / {} / {} / $ {}".format(top20Points.title.values[i], top20Points.country.values[i], top20Points.province.values[i], top20Points.price.values[i]))
    print("-----------------------------------------------------------------------------------------------------------------------")
labels = top20Points.country.value_counts().index
values = top20Points.country.value_counts().values

trace = go.Pie(labels=labels, values=values)

iplot([trace])
df.variety.value_counts()

fig = {
  "data": [
    {
      "values": df.variety.value_counts().values[:10],
      "labels": df.variety.value_counts().index[:10],
      "name": "Variaty",
      "hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    ],
  "layout": {
        "title":"Variaty",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grapes",
                "x": 0.5,
                "y": 0.5
            },
        ]
    }
}

iplot(fig)
meanPoints = df.points.mean()
df["Above_Average"] = [1 if i > meanPoints else 0 for i in df.points]
# This process can takes long time. Because we have a lot of descriptions.

import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp

descriptionList = list()
lemma = nlp.WordNetLemmatizer()

for description in df.description:
    description = re.sub("[^a-zA-Z]"," ",description) # We use regular expression to delete non-alphabetic characters on data.
    description = description.lower() # Since upper and lower characters are (e.g a - A) evaluated like they are different each other by computer we make turn whole characters into lowercase.
    description = nltk.word_tokenize(description) # We tokenized the statement
    description = [i for i in description if not i in set(stopwords.words("english"))] # We will remove words like 'the', 'or', 'and', 'is' etc.
    description = [lemma.lemmatize(i)for i in description] # e.g: loved => love
    description = " ".join(description) # Now we turn our words list into sentence again
    descriptionList.append(description)
from sklearn.feature_extraction.text import CountVectorizer
# We try to use most common 1500 words to make a prediction.

max_features = 1500
count_vectorizer = CountVectorizer(max_features=max_features) # stop_words="english" i istersek burada yazabilirdik, lowercase' de burada kullanabilirdik vs.
sparce_matrix = count_vectorizer.fit_transform(descriptionList)
sparce_matrix = sparce_matrix.toarray()
print("Most Frequent {} Words: {}".format(max_features, count_vectorizer.get_feature_names()))
x = sparce_matrix
y = df.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
# Prediction
y_pred = nb.predict(x_test)
print("Accuracy: {:.2f}%".format(nb.score(y_pred.reshape(-1,1), y_test)*100))