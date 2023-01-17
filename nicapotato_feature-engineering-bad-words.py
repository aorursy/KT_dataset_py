import time
import datetime
start = time.time()

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

print("Datasets Used:")
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv').fillna(' ')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv').fillna(' ')
badwords = pd.read_csv('../input/bad-bad-words/bad-words.csv', header=None).iloc[:,0].tolist()

print("Data Shape:")
print("Train Shape: {} Rows, {} Columns".format(*df.shape))
print("Test Shape: {} Rows, {} Columns".format(*test.shape))
print("Bad Words Shape: lenght {}".format(len(badwords)))
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_text = df['comment_text']
test_text = test['comment_text']
all_text = pd.DataFrame(pd.concat([train_text, test_text]))
print("All Data Shape: {} Text Rows".format(all_text.shape))
# Glance
print(badwords[:10])
df.head()
%%time
df["badwordcount"] = df['comment_text'].apply(
    lambda comment: sum(comment.count(w) for w in badwords))
df['num_words'] = df['comment_text'].apply(
        lambda comment: len(comment.split()))
df['num_chars'] = df['comment_text'].apply(len)
df["normchar_badwords"] = df["badwordcount"]/df['num_chars']
df["normword_badwords"] = df["badwordcount"]/df['num_words']
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
f, ax = plt.subplots(figsize= [10,7])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
ax.set_title("Correlation Matrix for Toxity and New Features")
plt.show()
