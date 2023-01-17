# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



# Import required packages

import numpy as np 

import pandas as pd 

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





# Input data files are available in the "../input/" directory.

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Load the files into a pandas dataframe

df_train = pd.read_csv("../input/train.csv")

df_test  = pd.read_csv("../input/test.csv")
print("train.csv")

print("\t Number of Rows \t: ",df_train.shape[0])

print("\t Number of Columns \t: ",df_train.shape[1])



print("test.csv")

print("\t Number of Rows \t: ",df_test.shape[0])

print("\t Number of Columns \t: ",df_test.shape[1])

print("train.csv")

print("\t Column Names \t: ",df_train.columns)

print("\n")

print("test.csv")

print("\t Column Names \t: ",df_test.columns)
df_train.head()
df_test.head()
# Targer Variable count

is_dup = df_train['is_duplicate'].value_counts()

is_dup
# Target Variable percentage in train

df_train['is_duplicate'].value_counts() / df_train['is_duplicate'].count()
# visualizing target variable

sns.barplot(is_dup.index, is_dup.values, color='lightgreen')
# All qid's from train

train_qid = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

# Number of Unique qid's

len(np.unique(train_qid))
stops = set(stopwords.words('english'))
def likeness(data):

    like_words = []

    # get the questions

    q1 = str(data['question1']).lower()

    q2 = str(data['question2']).lower()

    # split question string into tokens/words

    t1 = word_tokenize(q1)

    t2 = word_tokenize(q2)

    common_words = list(set(t1) & set(t2))

    # remove stop words

    for w in common_words:

        if w not in stops:

            like_words.append(w)

    all_words = list(set(t1) | set(t2))    

    like_percentage = len(like_words) / len(all_words)

    return like_percentage
# add traget column in test

df_test['is_duplicate'] = df_test.apply(likeness, axis=1)
df_test.head()
submission = pd.DataFrame(df_test, columns=['test_id','is_duplicate'])

submission.to_csv('submission.csv', index=False)