import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv("../input/unigram_freq.csv")
df.head(10)
top10 = df.iloc[0:10]
plt.figure(figsize=(10,6))
sns.barplot("word", "count", data=top10, palette="Blues_d").set_title("Top 10 Words")
df.sort_values(by="count").iloc[0:10]
s = df.word.str.len().sort_values(ascending=False).index
longest10 = df.reindex(s).iloc[0:10]
plt.figure(figsize=(10,6))
sns.barplot("count", "word", data=longest10, orient="h", palette="Blues_d").set_title("Top 10 Longest Words")
alphabet = df.reindex(s).iloc[::-1][2:28].sort_values(by="count", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot("word", "count", data=alphabet, palette="Blues_d").set_title("Alphabets")