import numpy as np

import pandas as pd

from IPython.display import display

import matplotlib.pyplot as plt



import nltk

nltk.download("wordnet")
data_dir = "../input/reddit-rcryptocurrency/"

datasets = {"%s%s.csv" % (data_dir, t) for t in ["2017-11", "2017-12", "2018-01", "2018-02", "2018-03"]}

dt_cols = {"author": str, "body": str, "created_utc": str}

df = pd.concat([pd.read_csv(dataset, usecols=dt_cols, dtype=dt_cols) for dataset in datasets], ignore_index=True)
df = df.dropna()

df.count()
df.head(n=10)
for txt in list(df["body"][:100]):

    print(txt)

    print("-----------------")
def words_num(txt):

     return len([x for x in txt.split() if 3 < len(x) < 30])



def plot_chars_and_words():

    plt.figure(figsize=(15, 5))



    plt.subplot(121)

    plt.plot(list(df["body"].apply(len).sort_values()))

    plt.ylabel("length of text")



    plt.subplot(122)

    plt.plot(list(df["body"].apply(lambda txt: words_num(txt)).sort_values()))

    plt.ylabel("number of words")



    plt.show()



plot_chars_and_words()
conf_min_chars = 150

conf_max_chars = 2000

conf_min_words = 20

conf_max_words = 300



condition = lambda txt: conf_max_chars > len(txt) > conf_min_chars and conf_max_words > words_num(txt) > conf_min_words

df = df[df["body"].map(condition)]

df.count()
plot_chars_and_words()
def plot_posts_frequency():

    plt.figure(figsize=(13, 5))

    plt.plot(df["author"].value_counts().to_list())

    plt.ylabel("posts frequency")

    plt.show()



plot_posts_frequency()
conf_min_posts_num = 200

conf_max_posts_num = 1000

df = df[df.groupby("author")["author"].transform("size") > conf_min_posts_num]

df = df[df.groupby("author")["author"].transform("size") < conf_max_posts_num]

"Total authors:", df["author"].nunique()
plot_posts_frequency()
conf_apply_stemming = True

conf_apply_lemmatizing = True

conf_to_lower_case = True



w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

lem = nltk.stem.wordnet.WordNetLemmatizer()

stem = nltk.stem.porter.PorterStemmer()



def tokenize(text):

    res = ""

    for w in w_tokenizer.tokenize(text):

        if conf_apply_lemmatizing:

            w = lem.lemmatize(w)

        if conf_apply_stemming:

            w = stem.stem(w)

        if conf_to_lower_case:

            w = w.lower()

        res += w + " "

    return res.strip()



df["optimized_body"] = df["body"].apply(tokenize)
# pick a target author and train a model to recognize what he wrote

df["author"]
# pick target author and add a new column which indicates whether or not he is the target author.

# this is the column that we're going to try to predict.

target_author = "simmol"

df["is_target"] = df["author"]==target_author

df.head()
df["is_target"].value_counts()
sdf = pd.concat([df[df["is_target"] == True], df[df["is_target"] == False].sample(n=400)])

sdf = sdf.sample(frac=1).reset_index(drop=True) # random shuffle

sdf.head()
from sklearn.feature_extraction.text import TfidfVectorizer



conf_max_df=.80

conf_min_df=0.01

conf_n_gram_range=(1,3)

conf_remove_stop_words=False



vectorizer = TfidfVectorizer(

    max_df=conf_max_df,

    min_df=conf_min_df,

    ngram_range=conf_n_gram_range,

    stop_words="english" if conf_remove_stop_words else None,

    token_pattern=r'(?u)\b[A-Za-z0-9]+\b'

)



vectors = vectorizer.fit_transform(sdf["optimized_body"])

vectors.shape
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    vectors,

    sdf["is_target"],

    test_size=0.20,

    random_state=42

)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import classification_report



classifier = SGDClassifier(

    loss="log",

    penalty="l1",

    max_iter=15,

    n_jobs=-1,

    random_state=42

)



classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions, target_names=["0", "1"]))
from sklearn.svm import LinearSVC



classifier = LinearSVC(tol=1e-8, max_iter=10000)



classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)



print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions, target_names=["0", "1"]))
from sklearn.naive_bayes import GaussianNB



classifier = GaussianNB()



classifier.fit(X_train.toarray(), y_train)

predictions = classifier.predict(X_test.toarray())



print(accuracy_score(y_test, predictions))

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions, target_names=["0", "1"]))
y_true = np.array([])

y_pred = np.array([])



for target_author, _ in df["author"].value_counts().items():

    classifier = LinearSVC(tol=1e-9, max_iter=1000000)

    

    df["is_target"] = df["author"]==target_author

    dft = df[df["is_target"] == True]    

    dff = df[df["is_target"] == False].sample(n=dft.shape[0])

    sdf = pd.concat([dft, dff])

    sdf = sdf.sample(frac=1).reset_index(drop=True) # random shuffle

    

    vectors = vectorizer.fit_transform(sdf["optimized_body"])

    X_train, X_test, y_train, y_test = train_test_split(

        vectors,

        sdf["is_target"],

        test_size=0.20,

        random_state=42

    )

    classifier.fit(X_train.toarray(), y_train)

    predictions = classifier.predict(X_test.toarray())



    y_true = np.append(y_true, y_test)

    y_pred = np.append(y_pred, predictions)

    

print(accuracy_score(y_true, y_pred))

print(confusion_matrix(y_true, y_pred))

print(classification_report(y_true, y_pred, target_names=["0", "1"]))