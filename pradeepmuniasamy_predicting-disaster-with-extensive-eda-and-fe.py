## Essential Python Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#data Visualization

import matplotlib.pyplot as plt 

import seaborn as sns 



#Feature Engineering 

import nltk

import string

from nltk.corpus import stopwords

import string

import gc

eng_stopwords = set(stopwords.words("english"))

pd.options.mode.chained_assignment = None



color = sns.color_palette()

%matplotlib inline
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df=pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sample_df=pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print("Number of rows in train dataset:",train_df.shape[0])

print("Number of rows test dataset:",test_df.shape[0])
train_df.isnull().sum()
test_df.isnull().sum()
train_df.head()
cnt_target = train_df['target'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(cnt_target.index, cnt_target.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Target', fontsize=12)

plt.show()
grouped_df = train_df.groupby('target')

for target_type, group in grouped_df:

    print("Target type:",target_type )

    cnt = 0

    for ind, row in group.iterrows():

        print(row["text"])

        cnt += 1

        if cnt == 5:

            break

    print("\n")
df=train_df.append(test_df) ##we have appended the train and test dataset into a single dataframe

print("Missing Target values:",(df.target.isnull().sum())) 
##Number of words in the text

df["num_words"] = df["text"].apply(lambda x : len(str(x).split()))



##Number of Unique words in the text

df["num_unique_words"]=df["text"].apply(lambda x: len(set(str(x).split())))



##Number of Characters in the text

df["num_characters"]=df["text"].apply(lambda x : len(str(x)))



##Number of Stopwords in the text

df["num_stopwords"]=df["text"].apply(lambda x : len([w for w in str(x).lower().split() if w in eng_stopwords]))



##Number of Punctuations in the text

df["num_punctuations"]=df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))





##Number of words in Upper case

df["num_uppercase"]=df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



##Number of words in Title case

df["num_titlecase"]=df["text"].apply(lambda x:len([w for w in str(x).split() if w.istitle()]))



#Average words of each words

df["avg_words"]=df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_df=df[df["target"].isnull()!=True]

test_df=df[df["target"].isnull()==True]

test_df.drop("target",axis=1,inplace=True)



del df

gc.collect()
plt.figure(figsize=(12,8))

sns.violinplot(x='target', y='num_words', data=train_df)

plt.xlabel('target', fontsize=12)

plt.ylabel('Number of words in text', fontsize=12)

plt.title("Number of words by type", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.violinplot(x='target', y='num_punctuations', data=train_df)

plt.xlabel('target', fontsize=12)

plt.ylabel('Number of punctuations in text', fontsize=12)

plt.title("Number of punctuations by type", fontsize=15)

plt.show()