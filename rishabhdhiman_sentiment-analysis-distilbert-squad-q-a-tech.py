!mkdir data
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

import string

from wordcloud import WordCloud, STOPWORDS

import tensorflow as tf

import missingno as msno

from collections import defaultdict

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import json



%matplotlib inline
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

sample_submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")
train.shape, test.shape
# Finding the missing values

fig, axes = plt.subplots(1, 2, figsize=(15,6))

plt.tight_layout(pad = 10)

msno.bar(train, ax = axes[0])

msno.bar(test, ax = axes[1])
print(train[train["text"].isnull() == True])

# We can drop this row

train.dropna(inplace = True)
fig, axes = plt.subplots(1, 2, figsize=(15,6))

plt.tight_layout(pad = 10)

msno.bar(train, ax = axes[0])

msno.bar(test, ax = axes[1])
# Distribution of sentiment class

fig, axes = plt.subplots(1, 2, figsize=(15,8))

fig.suptitle("Comparing Ratio of Neutral Negative and Positive in train and test data", fontsize = 25)

plt.tight_layout(pad = 3.5)

sns.countplot(x = "sentiment", data = train, ax = axes[0])

sns.countplot(x = "sentiment", data = test, ax = axes[1])

axes[0].set_xlabel("Sentiment", fontsize = 20)

axes[0].set_ylabel("Count", fontsize = 20)

axes[1].set_xlabel("Sentiment", fontsize = 20)

axes[1].set_ylabel("Count", fontsize = 20)

sns.despine()
# Percentage of neutral, negative, positive words in train and test data

def pert_count(data, category):

    return (len(data[data["sentiment"] == category])/len(data)) * 100



print(f"Percentage of neutral words in train --> {pert_count(train, 'neutral')}%")

print(f"Percentage of negative words in train --> {pert_count(train, 'negative')}%")

print(f"Percentage of positive words in train --> {pert_count(train, 'positive')}%")

print(f"Percentage of neutral words in test --> {pert_count(test, 'neutral')}%")

print(f"Percentage of negative words in test --> {pert_count(test, 'negative')}%")

print(f"Percentage of positive words in test --> {pert_count(test, 'positive')}%")
# Length of words for in each category

def len_sent(data):

    return len(data.split())



train["num_words_text"] = train["text"].apply(lambda x : len_sent(x))

test["num_words_text"] = test["text"].apply(lambda x : len_sent(x))

train["num_words_selected_text"] = train["selected_text"].apply(lambda x : len_sent(x))
fig, axes = plt.subplots(3, 1, sharey = True, figsize = (15, 20))

fig.suptitle("Comparing train and test data based on sentiment length", fontsize = 25)

sns.kdeplot(train[train["sentiment"] == "neutral"]["num_words_text"].values, ax  = axes[0], shade = True, color = "blue", label = "train")

sns.kdeplot(test[test["sentiment"] == "neutral"]["num_words_text"].values, ax  = axes[0], shade = True, color = "red", label = "test")

sns.kdeplot(train[train["sentiment"] == "negative"]["num_words_text"].values, ax  = axes[1], shade = True, color = "blue", label = "train")

sns.kdeplot(test[test["sentiment"] == "negative"]["num_words_text"].values, ax  = axes[1], shade = True, color = "red", label = "test")

sns.kdeplot(train[train["sentiment"] == "positive"]["num_words_text"].values, ax  = axes[2], shade = True, color = "blue", label = "train")

sns.kdeplot(test[test["sentiment"] == "positive"]["num_words_text"].values, ax  = axes[2], shade = True, color = "red", label = "test")

axes[0].set_xlabel("Sentiment_length_neutral", fontsize = 20)

axes[0].set_ylabel("Distribution", fontsize = 20)

axes[1].set_xlabel("Sentiment_length_negative", fontsize = 20)

axes[1].set_ylabel("Distribution", fontsize = 20)

axes[2].set_xlabel("Sentiment_length_positive", fontsize = 20)

axes[2].set_ylabel("Distribution", fontsize = 20)

plt.legend()
# Comparing test and selected text column

fig, axes = plt.subplots(3, 1, sharey = True, figsize = (15, 20))

fig.suptitle("", fontsize = 25)

sns.kdeplot(train[train["sentiment"] == "neutral"]["num_words_text"].values, ax  = axes[0], shade = True, color = "blue", label = "text")

sns.kdeplot(train[train["sentiment"] == "neutral"]["num_words_selected_text"].values, ax  = axes[0], shade = True, color = "red", label = "selected_text")

sns.kdeplot(train[train["sentiment"] == "negative"]["num_words_text"].values, ax  = axes[1], shade = True, color = "blue", label = "text")

sns.kdeplot(train[train["sentiment"] == "negative"]["num_words_selected_text"].values, ax  = axes[1], shade = True, color = "red", label = "selected_text")

sns.kdeplot(train[train["sentiment"] == "positive"]["num_words_text"].values, ax  = axes[2], shade = True, color = "blue", label = "text")

sns.kdeplot(train[train["sentiment"] == "positive"]["num_words_selected_text"].values, ax  = axes[2], shade = True, color = "red", label = "selected_text")

axes[0].set_xlabel("Sentiment_length_neutral", fontsize = 20)

axes[0].set_ylabel("Distribution", fontsize = 20)

axes[1].set_xlabel("Sentiment_length_negative", fontsize = 20)

axes[1].set_ylabel("Distribution", fontsize = 20)

axes[2].set_xlabel("Sentiment_length_positive", fontsize = 20)

axes[2].set_ylabel("Distribution", fontsize = 20)

plt.legend()
train["text"] = train["text"].apply(lambda x : x.strip())

train["selected_text"] = train["selected_text"].apply(lambda x : x.strip())





train["is_equal"] = (train["text"] == train["selected_text"])

df_neutral = train[train["sentiment"] == "neutral"]

percentage = (len(df_neutral[df_neutral["is_equal"] == True])/len(df_neutral)) * 100

print(f"Percentage of text column sentences is equal selected_text column for neutral sentiment --> {percentage}")
# Punctuation count in train["text"], train["selected_text"]

def punc_count(data):

    return len([w for w in data if w in string.punctuation])



train["text_punc_count"] = train["text"].apply(lambda x : punc_count(x))

train["selected_text_punc_count"] = train["selected_text"].apply(lambda x : punc_count(x))
plt.figure(figsize = (15, 8))

sns.kdeplot(train["text_punc_count"].values, shade = True, color = "blue", label = "Text punc count")

sns.kdeplot(train["selected_text_punc_count"].values, shade = True, color = "yellow", label = "Selected text punc count")

plt.title("Punctuation Count", fontsize = 30)

plt.xlabel("Punctuation Count", fontsize = 20)

plt.ylabel("Distribution", fontsize = 20)

sns.despine()

plt.legend(loc = "lower right")
# Most repeated words in text column and selected_text

stopwords = set(STOPWORDS)

def word_cloud(data, title):

    wordcloud = WordCloud(

    background_color = "black",

    max_font_size = 40,

    max_words = 200,

    stopwords = stopwords,

    scale = 3).generate(str(data))

    fig = plt.figure(figsize = (15, 15))

    plt.axis("off")

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.25)



    plt.imshow(wordcloud)

    plt.show()
word_cloud(train["text"], "Most Repeated words in train['text']")
word_cloud(train[train["sentiment"] == "neutral"]["text"], "Most Repeated words in neural sentences train['text']")
word_cloud(train[train["sentiment"] == "negative"]["text"], "Most Repeated words in negative sentences train['text']")
word_cloud(train[train["sentiment"] == "positive"]["text"], "Most Repeated words in positive sentences train['text']")
# N-Grams for neutral, positive negative sentences

def n_grams(ngram, data):

    freq_dict = defaultdict(int)

    for text in data:

        tokens = [w for w in text.lower().split() if w != " " if w not in stopwords]

        ngrams = zip(*[tokens[i:] for i in range(ngram)])

        list_grams = [" ".join(ngram) for ngram in ngrams]

        for word in list_grams:

            freq_dict[word] += 1

    fd_sorted =  pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])   

    fd_sorted.columns = ["word", "wordcount"]

    return fd_sorted

                    

fd_sorted_neutral1 = n_grams(1, train[train["sentiment"] == "neutral"]["text"])    

fd_sorted_negative1 = n_grams(1, train[train["sentiment"] == "negative"]["text"])    

fd_sorted_positive1 = n_grams(1, train[train["sentiment"] == "positive"]["text"]) 



fd_sorted_neutral2 = n_grams(2, train[train["sentiment"] == "neutral"]["text"])    

fd_sorted_negative2 = n_grams(2, train[train["sentiment"] == "negative"]["text"])    

fd_sorted_positive2 = n_grams(2, train[train["sentiment"] == "positive"]["text"]) 



fd_sorted_neutral3 = n_grams(3, train[train["sentiment"] == "neutral"]["text"])    

fd_sorted_negative3 = n_grams(3, train[train["sentiment"] == "negative"]["text"])    

fd_sorted_positive3 = n_grams(3, train[train["sentiment"] == "positive"]["text"])
fig, axes = plt.subplots(3, 1, figsize = (10, 18))

plt.tight_layout(pad = 7.5)

plt.suptitle("Unigrams", fontsize = 25)

sns.despine()

l = ["neutral", "negative", "positive"]

for i in range(3):

    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted_" + str(l[i]) + str(1)].iloc[:20, :], ax = axes[i])

    axes[i].set_title(f"Most repeated words in {l[i]} sentences", fontsize = 20)

    axes[i].set_xlabel("WordCount", fontsize = 15)    

    axes[i].set_ylabel("Word", fontsize = 15)
fig, axes = plt.subplots(3, 1, figsize = (10, 18))

plt.tight_layout(pad = 7.5)

plt.suptitle("Bigrams", fontsize = 25)

sns.despine()

l = ["neutral", "negative", "positive"]

for i in range(3):

    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted_" + str(l[i]) + str(2)].iloc[:20, :], ax = axes[i])

    axes[i].set_title(f"Most repeated words in {l[i]} sentences", fontsize = 20)

    axes[i].set_xlabel("WordCount", fontsize = 15)    

    axes[i].set_ylabel("Word", fontsize = 15)
fig, axes = plt.subplots(3, 1, figsize = (10, 18))

plt.tight_layout(pad = 7.5)

plt.suptitle("Trigrams", fontsize = 25)

sns.despine()

l = ["neutral", "negative", "positive"]

for i in range(3):

    sns.barplot(x = "wordcount", y = "word", data = globals()["fd_sorted_" + str(l[i]) + str(3)].iloc[:20, :], ax = axes[i])

    axes[i].set_title(f"Most repeated words in {l[i]} sentences", fontsize = 20)

    axes[i].set_xlabel("WordCount", fontsize = 15)    

    axes[i].set_ylabel("Word", fontsize = 15)
train_array = np.array(train.iloc[:, :4])

test_array = np.array(test.iloc[:, :3])

use_cuda = True
# Getting starting index of selected_text found in text

def start_index(text, selected_text):

    start_index = text.lower().find(selected_text.lower())

    l.append(start_index)

    

l = []

for i in range(len(train_array)):

    start_index(train_array[i, 1], train_array[i, 2])
# We are taking

# question --> sentiment

# context --> text

# answer --> selected_text



def quesa_format_train(train):

    out = []

    for i, row in enumerate(train):

        qas = []

        con = []

        ans = []

        question = row[-1]

        answer = row[2]

        context = row[1]

        qid = row[0]

        answer_start = l[i]

        ans.append({"answer_start": answer_start, "text": answer.lower()})

        qas.append({"question": question, "id": qid, "is_impossible": False, "answers": ans})

        out.append({"context": context.lower(), "qas": qas})



    return out

        

    

train_json_format = quesa_format_train(train_array)

with open('data/train.json', 'w') as outfile:

    json.dump(train_json_format, outfile)
# Similarly for text data



def quesa_format_test(train):

    out = []

    for i, row in enumerate(train):

        qas = []

        con = []

        ans = []

        question = row[-1]

#         answer = row[2]

        context = row[1]

        qid = row[0]

        answer_start = l[i]

        ans.append({"answer_start": 1000000, "text": "__None__"})

        qas.append({"question": question, "id": qid, "is_impossible": False, "answers": ans})

        out.append({"context": context.lower(), "qas": qas})

    return out

        

    

test_json_format = quesa_format_test(test_array)



with open('data/test.json', 'w') as outfile:

    json.dump(test_json_format, outfile)
!pip install '/kaggle/input/train-requirements/seqeval-0.0.12-py3-none-any.whl' -q

!pip install '/kaggle/input/train-requirements/simpletransformers-0.22.1-py3-none-any.whl' -q
from simpletransformers.question_answering import QuestionAnsweringModel



model_path = '/kaggle/input/transformers-pretrained-distilbert/distilbert-base-uncased-distilled-squad/'

# MODEL_PATH = QuestionAnsweringModel.from_pretrained('distilbert-base-uncased-distilled-squad'



# Create the QuestionAnsweringModel

model = QuestionAnsweringModel('distilbert', 

                               model_path, 

                               args={'reprocess_input_data': True,

                                     'overwrite_output_dir': True,

                                     'learning_rate': 5e-5,

                                     'num_train_epochs': 4,

                                     'max_seq_length': 128,

                                     'doc_stride': 64,

                                     'fp16': False,

                                    },

                              use_cuda=use_cuda)



model.train_model('data/train.json')
pred = model.predict(test_json_format)
df = pd.DataFrame.from_dict(pred)

sample_submission["selected_text"] = df["answer"]

# new_df = sample_submission.merge(test,how="inner",on="textID")

# new_df["selected_text"] = np.where((new_df["sentiment"] == "neutral"),new_df["text"], new_df["selected_text"])

# submission = new_df[["textID", "selected_text"]]

sample_submission.to_csv("submission.csv", index = False)

print("File submitted successfully.")