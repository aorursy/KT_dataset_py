import numpy as np 
import pandas as pd 
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/en_train.csv")
train.shape
train.head()
test = pd.read_csv("../input/en_test.csv")
test.shape
test.head()
sample_submission = pd.read_csv("../input/en_sample_submission.csv")
sample_submission.head()
test.shape[0]/(train.shape[0] + test.shape[0])
num_train_sentences = len(train.sentence_id.unique())
num_train_sentences
num_test_sentences = len(test.sentence_id.unique())
num_test_sentences
num_test_sentences / (num_train_sentences + num_test_sentences)
train_sentences = train.groupby("sentence_id")["sentence_id"].count()
train_sentences.describe()
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.set_style("whitegrid")
count_length_fig = sns.countplot(train_sentences, ax=ax)
for item in count_length_fig.get_xticklabels():
    item.set_rotation(90)
test_sentences = test.groupby("sentence_id")["sentence_id"].count()
test_sentences.describe()
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.set_style("whitegrid")
count_length_fig = sns.countplot(test_sentences, ax=ax)
for item in count_length_fig.get_xticklabels():
    item.set_rotation(90)
max_id = train_sentences[train_sentences == train_sentences.max()].index.values
max_id
long_example = train[train.sentence_id==max_id[0]].before.values.tolist()
long_example= ' '.join(long_example)
long_example
min_id = train_sentences[train_sentences == train_sentences.min()].index.values
min_id
for n in range(5):
    small_example = train[train.sentence_id==min_id[n]].before.values.tolist()
    small_example= ' '.join(small_example)
    print(small_example)
median_id = train_sentences[train_sentences == train_sentences.median()].index.values
median_id
for n in range(5):
    median_example = train[train.sentence_id==median_id[n]].before.values.tolist()
    median_example= ' '.join(median_example)
    print(median_example)
len(train.token_id.unique())
len(train["class"].unique())
fig, ax = plt.subplots(1,1, figsize=(10,12))
#sns.set_style("whitegrid")
count_classes_fig = sns.countplot(y="class", data=train, ax=ax)
for item in count_classes_fig.get_xticklabels():
    item.set_rotation(45)
train.groupby("class")["class"].count()
most_electronic_cases = train[train["class"]=='ELECTRONIC'].groupby("before")["before"].count(
).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_electronic_cases.index, y=most_electronic_cases.values)
most_verbatim_cases = train[train["class"]=='VERBATIM'].groupby("before")["before"].count(
).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_verbatim_cases.index, y=most_verbatim_cases.values)
len(train.before.unique())
train_word_counts = train.groupby("before")["before"].count().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=train_word_counts.index, y=train_word_counts.values)
len(test.before.unique())
test_word_counts = test.groupby("before")["before"].count().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=test_word_counts.index, y=test_word_counts.values)
train["change"] = 0
train.loc[train.before!=train.after, "change"] = 1
train["change"].value_counts()
most_changed_words = train[train.change==1].groupby("before")["before"].count(
).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(1,1,figsize=(15,5))
sns.barplot(x=most_changed_words.index, y=most_changed_words.values)
fig, ax = plt.subplots(1,1,figsize=(15,5))
changes_classes_fig = sns.countplot(x="class", data=train[train.change==1])
for item in changes_classes_fig.get_xticklabels():
    item.set_rotation(45)
unique_digits = set(train[train["class"]=="DIGIT"].before.unique().tolist())
unique_dates = set(train[train["class"]=="DATE"].before.unique().tolist())
overlap = unique_digits.intersection(unique_dates)
len(overlap)
list(overlap)[0:10]
train_sentences_info = pd.DataFrame(index=train.sentence_id.unique())
test_sentences_info = pd.DataFrame(index=test.sentence_id.unique())
train_sentences_info["length"] = train_sentences
test_sentences_info["length"] = test_sentences

train_sentences_info["num_changes"] = train.groupby("sentence_id")["change"].sum()
train_sentences_info.head()
train_sentences_info["num_changes"].describe()
train_sentences_info[train_sentences_info.num_changes==94]
most_changed_sentence_id = train_sentences_info[train_sentences_info.num_changes==94].index.values[0]
most_changed_sentence = train[train.sentence_id==most_changed_sentence_id].before.values.tolist()
most_changed_sentence = ' '.join(most_changed_sentence)
most_changed_sentence
most_changed_sentence_after = train[train.sentence_id==most_changed_sentence_id].after.values.tolist()
most_changed_sentence_after = ' '.join(most_changed_sentence_after)
most_changed_sentence_after
plt.figure(figsize=(15,5))
#sns.jointplot(x="length", y="num_changes", data=train_sentences_info, kind="kde")
sns.jointplot(x="length", y="num_changes", data=train_sentences_info)
plt.figure(figsize=(15,5))
sns.jointplot(x="length", y="num_changes", data=train_sentences_info[train_sentences_info.num_changes > 1])
plt.figure(figsize=(10,6))
sns.countplot(x="token_id", data=train[(train.change==1) & (train.token_id <=30)])
plt.xlabel("Token ID")
plt.ylabel("Number of changes")
collected = train[train.change==1][["sentence_id", "token_id"]]
collected["sentence_length"] = collected["sentence_id"].apply(lambda l: train_sentences_info.loc[l, "length"])
collected = collected[collected.sentence_length <= 30]
collected.head()
changed_positions = collected.groupby("sentence_length")["token_id"].value_counts().unstack()
changed_positions.describe()
changed_positions.fillna(0.0, inplace=True)
changed_positions = changed_positions.applymap(lambda l: np.log10(l+1))
mask = np.zeros_like(changed_positions.values)
mask[np.triu_indices_from(mask, k=2)] = True
plt.figure(figsize=(15,10))
sns.heatmap(changed_positions, mask=mask, cmap="magma")
plt.xlabel("position / token_id")
plt.ylabel("sentence length")
plt.title("Frequency of changed positions in sentences (log10-scale)")
train7_info = train_sentences_info[train_sentences_info.length==7]
train7_sentence_ids = train7_info[train7_info.num_changes > 0].index.values
train7 = train[train.sentence_id.isin(train7_sentence_ids)]
train7_pos4_examples = train7[(train7.token_id==4) & (train7.change==1)].iloc[0:10,:]
sentence_ids = train7_pos4_examples.sentence_id.values
for idx in sentence_ids:
    before_sentence = train7[train7.sentence_id==idx].before.values.tolist()
    after_sentence = train7[train7.sentence_id==idx].after.values.tolist()
    before_sentence = ' '.join(before_sentence)
    after_sentence = ' '.join(after_sentence)
    print("before:" + before_sentence + "\n" + "_____after:" + after_sentence)