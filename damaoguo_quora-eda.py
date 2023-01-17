# 加载软件包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#列名
print("train:",train_df.columns.values,"test:",test_df.columns.values)
train_sentences = train_df["question_text"].apply(lambda x: x.split()).values
test_sentences = test_df["question_text"].apply(lambda x: x.split()).values
print(train_sentences.shape,test_sentences.shape)

train_length = [len(sentence) for sentence in train_sentences]
test_length = [len(sentence) for sentence in test_sentences]
train_length_unique = [len(set(sentence)) for sentence in train_sentences]
test_length_unique = [len(set(sentence)) for sentence in test_sentences]
train_length_df = pd.Series(train_length)
test_length_df = pd.Series(test_length)
length1 = train_length_df.value_counts(sort=False)
length2 = test_length_df.value_counts(sort=False)
# 没有经过处理的train句子的长度
plt.figure(figsize=(12, 8))
length1.plot(kind='bar')
# 没有经过处理的test句子的长度
plt.figure(figsize=(12, 8))
length2.plot(kind='bar')
length1_unique = pd.Series(train_length_unique).value_counts(sort=False)
length2_unique = pd.Series(test_length_unique).value_counts(sort=False)
# 没有经过处理的train句子的长度
plt.figure(figsize=(12, 8))
length1_unique.plot(kind='bar')
# 没有经过处理的test句子的长度
plt.figure(figsize=(12, 8))
length2_unique.plot(kind='bar')