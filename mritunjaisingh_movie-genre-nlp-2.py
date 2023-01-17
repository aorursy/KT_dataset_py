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
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

%matplotlib inline
pd.set_option('display.max_colwidth', 300)
movie = pd.read_csv("/kaggle/input/movie-data/movie_metadata.csv", header=None)
movie.head()
movie.columns = ["movie_id", 1, "movie_name", 3, 4, 5 ,6, 7, "genre"]
movie.head(1)
plot = pd.read_csv("/kaggle/input/movie-data/plot_summaries.txt", sep = '\t', header=None)
plot.columns = ["movie_id", "plot"]
plot.head()
data = movie[["movie_id", "movie_name", "genre"]].merge(plot, on="movie_id")
data_test = pd.merge(plot, movie[["movie_id", "movie_name", "genre"]], on= "movie_id")
print(data.shape, data_test.shape)
data.head()
data[data["movie_name"] == "Narasimham"]
data["genre"][0]
list(json.loads(data["genre"][0]).values())
genre = []

for i in data["genre"]:
    #print(list(json.loads(i).values()))
    genre.append(list(json.loads(i).values()))
data["genre_new"] = genre
data.head(1)
data.dtypes
#type(str(data["genre_new"]))
data_new = data[~(data["genre_new"].str.len() == 0)]
data_new.shape
dumm = []

for i in genre:
    for j in i:
        #print(j)
        dumm.append(j)
all_genre = list(set(dumm))
var = sum(genre,[])
len(all_genre)
genre_new = nltk.FreqDist(dumm)
genre_new
type(genre_new)
len(genre_new.keys())
genre_df = pd.DataFrame.from_dict(genre_new, orient="index")
genre_df.columns = ["Count"]
genre_df.index.name = ["Genre"]
genre_df.head()
#genre_df["Genre"] = genre_df.index
#genre_df.reset_index()
del genre_df.index.name
genre_df = genre_df.reset_index()
genre_df.shape
genre_df.columns = ["Genre", "Count"]
genre_df.head(2)
plt.figure(figsize=(12,12))
sns.barplot(data=genre_df.sort_values("Count", ascending=False).loc[:20, :], x="Count", y="Genre")

def clean_text(text):
    
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = " ".join(text.split())
    text = text.lower()
    
    return text
data_new["clean_plot"] = data_new["plot"].apply(lambda x : clean_text(x))
data_new.head(2)
def freq_plot(text):
    
    words = " ".join([x for x in text])
    words = words.split()
    fdist = nltk.FreqDist(words)
    return fdist

fdist = freq_plot(data_new["clean_plot"])
words_df = pd.DataFrame.from_dict(fdist, orient="index")
words_df = words_df.reset_index()
words_df.columns = ["Word","Count"]
words_df.head()
plt.figure(figsize=(12,12))
sns.barplot(data= words_df.sort_values(by="Count",ascending= False).iloc[:20, :], x = "Count", y= "Word")
nltk.download("stopwords")
from nltk.corpus import stopwords
stopwords = set(stopwords.words("english"))
def remove_stopwords(text):
    no_stop = []
    
    for i in text.split():
        if i not in stopwords:
            no_stop.append(i)
    return " ".join(no_stop)
data_new["clean_plot"] = data_new["clean_plot"].apply(lambda x : remove_stopwords(x))
data_new.head(2)
from sklearn.preprocessing import MultiLabelBinarizer
multilabel_bina = MultiLabelBinarizer()
multilabel_bina.fit(data_new["genre_new"])
y = multilabel_bina.transform(data_new["genre_new"])
tfidf_vect = TfidfVectorizer(max_df= 0.8, max_features=10000)
data_new.shape
y.shape
xtrain, xval, ytrain, yval = train_test_split(data_new["clean_plot"], y, test_size = 0.2, random_state= 9)
tfidf_vect
xval.shape
xtrain_tfidf = tfidf_vect.fit_transform(xtrain)
xval_tfidf = tfidf_vect.transform(xval)
xtrain_tfidf.shape
xval_tfidf.shape
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,confusion_matrix
logistic_mod = LogisticRegression()
onevsall = OneVsRestClassifier(logistic_mod)
onevsall.fit(xtrain_tfidf, ytrain)
y_pred = onevsall.predict(xval_tfidf)
y_pred[2]
multilabel_bina.inverse_transform(y_pred)[34]
print(classification_report(yval, y_pred))
y_pred_prob = onevsall.predict_proba(xval_tfidf)
t = 0.3
y_pred_new = (y_pred_prob >= t).astype(int)
print(classification_report(yval, y_pred_new))
def new_val(x):
    
    x = clean_text(x)
    x = remove_stopwords(x)
    x_vec = tfidf_vect.transform([x])
    x_pred = onevsall.predict(x_vec)
    
    return multilabel_bina.inverse_transform(x_pred)
for i in range(5): 
  k = xval.sample(1).index[0] 
  print("Movie: ", data_new['movie_name'][k], "\nPredicted genre: ", new_val(xval[k])), print("Actual genre: ",data_new['genre_new'][k], "\n")
