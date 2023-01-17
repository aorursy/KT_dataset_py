
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split                # to split the data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score   
from sklearn.metrics import classification_report, confusion_matrix

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None

import os
print(os.listdir("../input"))

!ls ../input/
# List the embeddings provided by kaggle team
!ls ../input/embeddings/
## Read the train and test dataset and check the top few lines ##
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Number of rows in train dataset : ",train_df.shape[0])
print("Number of rows in test dataset : ",test_df.shape[0]) 
train_df.head()
#Check for the class-categorization count and also the class imbalance
cnt_srs = train_df['target'].value_counts()

plt.figure(figsize=(8,4))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('target', fontsize=12)
plt.show()
# Let us print some lines of each of the questions cagtegory in quora to try and understand their writing style if possible.
grouped_df = train_df.groupby('target')
for name, group in grouped_df:
    print("Target Name :", name)
    cnt =0
    for ind, row in group.iterrows():
        print(row['question_text'])
        cnt += 1
        if cnt == 2:
            break
    print("\n")
# Number of words in the text 
train_df["num_words"] = train_df["question_text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["question_text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["question_text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

train_df.head(3)
train_df.shape
train_df['num_words'].loc[train_df['num_words']>50] = 50 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_words', data=train_df)
plt.xlabel('target category', fontsize=12)
plt.ylabel('Number of words in text', fontsize=12)
plt.title("Number of words by target category", fontsize=15)
plt.show()
train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_punctuations', data=train_df)
plt.xlabel('target Name', fontsize=12)
plt.ylabel('Number of puntuations in text', fontsize=12)
plt.title("Number of punctuations by target category", fontsize=15)
plt.show()
train_df['num_chars'].loc[train_df['num_chars']>300] = 300 #truncation for better visuals
plt.figure(figsize=(10,6))
sns.boxplot(x='target', y='num_chars', data=train_df)
plt.xlabel('target Name', fontsize=12)
plt.ylabel('Number of characters in text', fontsize=12)
plt.title("Number of characters by target category", fontsize=15)
plt.show()