import os

import numpy as np 

import pandas as pd 



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
import nltk

from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords

from string import punctuation 

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore")
path = "../input"

df_train = pd.read_csv(os.path.join(path, "train.csv"))

df_test = pd.read_csv(os.path.join(path, "test.csv"))
df_train.shape
df_train.head()
df_test.shape
df_test.head()
df_train["target"].value_counts()
insincere = df_train[df_train["target"] == 1]

sincere = df_train[df_train["target"] == 0]
sincere.head()
insincere.head()
question_class = df_train["target"].value_counts()

question_class.plot(kind= "bar", color= ["blue", "orange"])

plt.title("Bar chart")

plt.show()
print(df_train["target"].value_counts())

print(sum(df_train["target"] == 1) / sum(df_train["target"] == 0) * 100, "percent of questions are insincere.")

print(100 - sum(df_train["target"] == 1) / sum(df_train["target"] == 0) * 100, "percent of questions are sincere")
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



# Generate a word cloud image

sincere_wordcloud = WordCloud(width=600, height=400).generate(str(sincere["question_text"]))

#Positive Word cloud

plt.figure(figsize=(10,8), facecolor="black")

plt.imshow(sincere_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show();
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



# Generate a word cloud image

insincere_wordcloud = WordCloud(width=600, height=400).generate(str(insincere["question_text"]))

#Positive Word cloud

plt.figure(figsize=(10,8), facecolor="black")

plt.imshow(insincere_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)

plt.show();
df_train["number_words"] = df_train["question_text"].apply(lambda x: len(x.split()))

df_test["number_words"] = df_test["question_text"].apply(lambda x: len(x.split()))
df_train["num_unique_words"] = df_train["question_text"].apply(lambda x: len(set(str(x).split())))

df_test["num_unique_words"] = df_test["question_text"].apply(lambda x: len(set(str(x).split())))
df_train["num_chars"] = df_train["question_text"].apply(lambda x: len(str(x)))

df_test["num_chars"] = df_test["question_text"].apply(lambda x: len(str(x)))
from nltk.corpus import stopwords 

stop_words = set(stopwords.words("english"))
df_train["num_stopwords"] = df_train["question_text"].apply(lambda x : len([nw for nw in str(x).split() if nw.lower() in stop_words]))

df_test["num_stopwords"] = df_test["question_text"].apply(lambda x : len([nw for nw in str(x).split() if nw.lower() in stop_words]))
df_train["num_punctuation"] = df_train["question_text"].apply(lambda x : len([np for np in str(x) if np in punctuation]))

df_test["num_punctuation"] = df_test["question_text"].apply(lambda x : len([np for np in str(x) if np in punctuation]))
df_train["num_uppercase"] = df_train["question_text"].apply(lambda x : len([nu for nu in str(x).split() if nu.isupper()]))

df_test["num_uppercase"] = df_test["question_text"].apply(lambda x : len([nu for nu in str(x).split() if nu.isupper()]))
df_train["num_lowercase"] = df_train["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.islower()]))

df_test["num_lowercase"] = df_test["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.islower()]))
df_train["num_title"] = df_train["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.istitle()]))

df_test["num_title"] = df_test["question_text"].apply(lambda x : len([nl for nl in str(x).split() if nl.istitle()]))
df_train[df_train["target"] == 1].describe()
df_train[df_train["target"] == 0].describe()
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="number_words", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of words", title="Box plot of number of words according to the target");
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="num_unique_words", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of unique words", title="Box plot of number of unique words according to the target");
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="num_chars", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of characters", title="Box plot of number of characters according to the target");
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="num_stopwords", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of stopwords", title="Box plot of number of stopwords according to the target");
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="num_lowercase", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of lowercases", title="Box plot of number of lowercases according to the target");
fig, ax = plt.subplots()

fig.set_size_inches(12, 10)

sns.boxplot(data=df_train, y="num_title", x="target",orient="v")

ax.set(xlabel="target", ylabel="number of titles", title="Box plot of number of titles according to the target");
def text_process(question):

    nopunc = [char for char in question if char not in punctuation]

    nopunc = "".join(nopunc)

    meaning = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

    return( " ".join( meaning ))
clean_train = df_train["question_text"].apply(text_process)
clean_train.head()
clean_test = df_test["question_text"].apply(text_process)
clean_test.head()
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(clean_train, df_train.target.values, test_size=0.2, stratify = df_train.target.values)
X_train.shape, X_val.shape
y_train.shape, y_val.shape
pipeline = Pipeline([("cv",CountVectorizer(analyzer="word",ngram_range=(1,4),max_df=0.9)),

                     ("clf",LogisticRegression(solver="saga", class_weight="balanced", C=0.45, max_iter=250, verbose=1))])
X_train, X_val, y_train, y_val = train_test_split(clean_train, df_train.target.values, test_size=0.1, stratify = df_train.target.values)
lr_model = pipeline.fit(X_train, y_train)
lr_model
y_pred = lr_model.predict(X_val)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_val, y_pred)

sns.heatmap(cm, cmap="Blues", annot=True, square=True, fmt=".0f");
print(classification_report(y_val, y_pred))
y_pred_final = pipeline.predict(clean_test)

y_pred_final
df_sub = pd.DataFrame({"qid":df_test["qid"], "prediction":y_pred_final})

df_sub.head()
df_sub.to_csv('submission.csv', index=False)