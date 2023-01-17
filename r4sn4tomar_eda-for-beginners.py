import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
#print information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
train_df.info()
#Generates descriptive statistics of a dataset, excluding NaN values.
train_df.describe()
train_df.head()
test_df.info()
test_df.describe()
test_df.head()
## Number of words in the comment_text ##
train_df["num_words"] = train_df["comment_text"].apply(lambda x: len(str(x).split()))
train_df.head()
#minimum length of a comment in train data
min(train_df.num_words)
#maximum length of comment
max(train_df.num_words)
#print rows where comment contains single token in train data
train_df[train_df.num_words==1][:3]
train_df[train_df.num_words==1411]
toxic_count = train_df.iloc[:,2:-1].sum(axis=0)
arr_toxic_count = toxic_count.values
index = toxic_count.index.values
#plot toxic count for each toxic class
import matplotlib.pyplot as plt
x_pos = [i for i,_ in enumerate(index)]
plt.figure(figsize=(5, 3))
plt.bar(x_pos,arr_toxic_count,align='edge')
plt.xlabel("Toxic Severity levels")
plt.ylabel("comments count")
plt.title("comment count of various toxic Severity levels")
plt.xticks(x_pos, index,rotation=90)

plt.show()
# column wise sum of toxic labels for each row
train_df['toxic_score'] = train_df.iloc[:,2:-2].sum(axis=1)
train_df.head()
train_df[train_df['toxic_score']>0][:5]
toxic_observation_count = len(train_df[train_df['toxic_score']>0])
non_toxic_observation_count = train_df.shape[0]-toxic_observation_count
toxicity_type = ["toxic", "non-toxic"]
comments_count= [toxic_observation_count, non_toxic_observation_count]
#plot toxic count for each toxic class
import matplotlib.pyplot as plt
x_pos = [i for i,_ in enumerate(toxicity_type)]
plt.figure(figsize=(3, 3))
plt.bar(x_pos,comments_count,align='edge')
plt.xlabel("toxicity Type")
plt.ylabel("comments count")
plt.title("comment count for toxic vs non-toxic")
plt.xticks(x_pos, toxicity_type,rotation=90)

plt.show()
