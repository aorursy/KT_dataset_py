import os

import collections



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from wordcloud import WordCloud

from nltk import sent_tokenize

from nltk.corpus import stopwords

from nltk import ngrams, FreqDist

from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
no_real_disater = 0

real_disaster = 1



no_real_disaster_label = "no_real_disaster"

real_disaster_label = "real_disaster"



target_names = ["Not real disaster", "Real disaster"]



target_dict = {

    "label": target_names,

    "value": [0, 1]

}

target_df = pd.DataFrame(target_dict)



original_columns = ["id", "keyword", "location", "text", "target"]



stop_words = stopwords.words('english')
input_folder = os.path.join("/kaggle", "input", "nlp-getting-started")

output_folder = os.path.join("/kaggle", "working")



train_file = os.path.join(os.path.join(input_folder, "train.csv"))

test_file = os.path.join(os.path.join(input_folder, "test.csv"))
train_df = pd.read_csv(train_file)

test_df = pd.read_csv(test_file)





# Adding target labels

train_df.loc[:, "target_str"] = pd.merge(train_df, target_df, how="left", left_on="target", right_on="value")["label"]



train_df.head()
print(f"Train size: {train_df.shape}")

print(f"Test size: {test_df.shape}")
train_df.info()
percent_missing = train_df.isnull().sum() * 100 / train_df.shape[0]

missing_value_df = pd.DataFrame({"percent_missing": percent_missing})



missing_value_df.sort_values(by="percent_missing", ascending=False).plot(

    kind="bar", title="NaNs by column in percentage", rot=45)
no_nan_df = train_df.groupby("target_str").count().transpose()



cond_disaster = train_df.target == real_disaster

nan_df = pd.concat([train_df[cond_disaster].isnull().sum(),

                    train_df[~cond_disaster].isnull().sum()], axis=1)

nan_df.columns = [real_disaster_label, no_real_disaster_label]

nan_df.sort_values(by=[real_disaster_label], ascending=False, inplace=True)



fig = plt.figure(figsize=(12, 5))



# Divide the figure into a 1x2 grid, and give me the first section

ax1 = fig.add_subplot(121)

# Divide the figure into a 1x2 grid, and give me the second section

ax2 = fig.add_subplot(122)



nan_df.plot(kind="bar", stacked=True, title="NaNs for columns by target", ax=ax1, rot=45)

no_nan_df.plot(kind="bar", stacked=True, title="NO NaNs for columns by target", ax=ax2, rot=45)
train_df.groupby("keyword").size().sort_values(ascending=False)[:30].plot(

    kind="bar", figsize=(15, 8), width=0.8, title="Top 25 keywords", rot=45)
train_df.groupby("keyword").size().sort_values(ascending=False)[-25:].plot(

    kind="bar", figsize=(15, 8), width=0.8, title="Last 25 keywords", rot=45)
train_df.groupby("location").size().sort_values(ascending=False)[:25].plot(

    kind="bar", figsize=(15, 8), width=0.8, title="Top 25 locations", rot=45)
train_df.groupby("location").size().sort_values(ascending=False)[-25:].plot(

    kind="bar", figsize=(15, 8), width=0.8, title="Last 25 locations", rot=45)
fig = plt.figure(figsize=(12, 8))



ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



train_df.groupby("target_str").size().sort_values(ascending=False).plot(

    kind="bar", width=0.8, title="Target", ax=ax1, rot=45)



train_df.groupby("target_str").size().sort_values(ascending=False).plot(

    kind="pie", title="Target", ax=ax2, legend=True, autopct="%.2f%%", labels=None)



ax2.axis("off")
for tweet, keyword, label in train_df[["text", "keyword", "target_str"]].sample(n=5).values:

    print(f"{keyword}\n{tweet}\n{label}\n")
# Adding new feature: text_lenght

train_df.loc[:, "text_lenght"] = train_df.text.str.len()
top_len_df = train_df[["text_lenght", "target", "target_str"]].sort_values(

    by="text_lenght", ascending=False).head(30).reset_index(drop=True)

last_len_df = train_df[["text_lenght", "target", "target_str"]].sort_values(

    by="text_lenght", ascending=False).tail(30).reset_index(drop=True)



fig, ax = plt.subplots(figsize=(15, 6))

top_len_df.plot(kind="bar", y="text_lenght", x="target_str", width=0.6, 

                ax=ax, title="Top 30 Tweet lenght", rot=45)
fig, ax = plt.subplots(figsize=(15, 5))

last_len_df.plot(kind="bar", y="text_lenght", x="target_str", width=0.6, 

                ax=ax, title="Last 30 Tweet lenght", rot=45)
def generate_word_frequences_dict(text_list, stop_words):

    # Filter stop words and remove puntuation

    word_list = [word for text in text_list for word in text if word.isalpha() and word not in stop_words]

    return pd.Series(word_list).value_counts().to_dict()



def generate_word_cloud(text_list, title, stop_words=stop_words):

    frequences_dict = generate_word_frequences_dict(text_list, stop_words)

    # Create and generate a word cloud image:

    wordcloud = WordCloud().generate_from_frequencies(frequences_dict)



    fig = plt.subplots(figsize=(20, 12))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.title(title, fontsize=32)

    plt.show()
def split_text(text, stop_words):

    # Split text into tokens

    tokens = word_tokenize(text)

    # Remove puntuations

    words = [word for word in tokens if word.isalpha()]

    # Filter stop words

    words = [word for word in words if word not in stop_words]

    

    return words
train_df.loc[:, "splited_text"] = train_df["text"].apply(split_text, args=(stop_words,))



generate_word_cloud(train_df[cond_disaster]["splited_text"].values.tolist(), "Wordcloud by real disaster tweets")

generate_word_cloud(train_df[~cond_disaster]["splited_text"].values.tolist(), "Wordcloud by NO real disaster tweets")

generate_word_cloud(train_df["splited_text"].values.tolist(), "Wordcloud by tweets")
train_df.drop("id", axis=1, inplace=True)
train_df.head()
train_df.loc[:, ["keyword", "location"]] = train_df[["keyword", "location"]].fillna("")
train_df.keyword.str.len().astype(bool)
train_df.loc[:, "with_keyword"] = train_df.keyword.str.len().astype(bool)

train_df.loc[:, "with_location"] = train_df.location.str.len().astype(bool)



train_df.head()
def text_cleaning(text, stop_words=stop_words):

    # Split text into tokens

    tokens = word_tokenize(text)

    # Remove puntuations

    words = [word for word in tokens if word.isalpha()]

    # Filter stop words

    words = [word for word in words if word not in stop_words]

    # Stemming

    porter = PorterStemmer()

    stemmed = [porter.stem(word) for word in words]

    

    return stemmed
train_df.loc[:, "splitted_cleaned_text"] = train_df["text"].apply(text_cleaning, args=(stop_words,))
for tweet, cleaned, label in train_df[["text", "splitted_cleaned_text", "target_str"]].sample(n=5).values:

    print(f"{tweet}\n{cleaned}\n{label}\n")
pipeline = Pipeline([

    ("bow", CountVectorizer(analyzer=text_cleaning)),  # strings to token integer counts

    ("tfidf", TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ("classifier", MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
x_train, x_test, y_train, y_test = train_test_split(train_df["text"], train_df["target"], test_size=0.20)

pipeline.fit(x_train, y_train)



y_pred = pipeline.predict(x_test)
print(classification_report(y_pred, y_test, target_names=target_names))

print(f"Confusion matrix:\n{confusion_matrix(y_pred, y_test)}")

print(f"\nAccuracy: {accuracy_score(y_pred, y_test)}")
y_pred = pipeline.predict(test_df["text"])
def create_submission_file(predictions, ids, path, filename="submission.csv"):

    submission_data = {

        "id": ids,

        "target": predictions

    }



    submission_df = pd.DataFrame(submission_data)

    submission_df.to_csv(os.path.join(path, filename), index=False)

    

    print("Good luck!")
create_submission_file(y_pred, test_df.id, output_folder)