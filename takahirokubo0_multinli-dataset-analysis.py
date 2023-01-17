import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
nlp = spacy.load("en_core_web_sm", parser=False, entity=False)
print(os.listdir("../input"))
df = pd.read_json("../input/multinli_0.9_test_matched_unlabeled.jsonl", lines=True)
df.head(5)
data = pd.concat([df[c] for c in ["genre", "sentence1"]], axis=1).reset_index()
data.head(5)
data["genre"].value_counts().plot(kind="bar")
data["length"] = data["sentence1"].apply(len)
data["length"].plot.hist(alpha=0.5, bins=20)
columns = data.groupby("genre").groups.keys()
def filter_and_measure_length(d, c):
    lengths = d[d["genre"] == c]["length"]
    lengths = lengths.rename(c).reset_index(drop=True)
    return lengths

pd.concat([filter_and_measure_length(data, c) for c in columns], axis=1).plot.hist(alpha=0.5, bins=20)
data.sort_values("length")["sentence1"].tail(3).values
data.sort_values("length")["sentence1"].head(3).values
duplicates = data[data.duplicated(["sentence1"], keep=False)].groupby("sentence1").groups.keys()
print(len(duplicates))
data_except_d = data.drop_duplicates(["sentence1"])
print("Without duplicate, data size becomes {} => {}".format(len(data), len(data_except_d)))
data_except_d["genre"].value_counts().plot(kind="bar")
data_except_d.sort_values("length")["sentence1"].tail(3).values
data_except_d.sort_values("length")["sentence1"].head(3).values
def get_middle(df, window=3):
    center = int(len(df) / 2)
    middle = df.iloc[center-window:center+window]
    return middle.head(window * 2)
get_middle(data_except_d)["sentence1"].values
# Basic spaCy 
print([t.text for t in nlp("we'll see")])
data_except_d["word_count"] = data_except_d["sentence1"].apply(lambda x: len(nlp(x)))
data_except_d["word_count"].plot.hist(bins=10)
data_filtered = data_except_d[(3 <= data_except_d["word_count"]) & (data_except_d["word_count"] <= 25)]
print("Extract appropriate length sentences, data size becomes {} => {}".format(len(data_except_d), len(data_filtered)))
data_filtered["genre"].value_counts().plot(kind="bar")
data_filtered.sort_values("length")["sentence1"].tail(3).values
data_filtered.sort_values("length")["sentence1"].head(5).values
data_filtered["genre"].value_counts()
data_selected = data_filtered.groupby("genre").apply(lambda x: x.sample(n=379)).drop(columns=["genre", "index"]).reset_index()
data_selected.head(5)
data_selected["genre"].value_counts().plot(kind="bar")
pd.concat([filter_and_measure_length(data_selected, c) for c in columns], axis=1).plot.hist(alpha=0.5, bins=20)