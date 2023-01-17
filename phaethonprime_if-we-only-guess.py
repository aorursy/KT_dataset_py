import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv('../input/JEOPARDY_CSV.csv')
grpd = df.groupby(" Category")

print("{:>30s}  {:4s}  {:4s}  {:<30s}".format("Category", "#Cat", "#Ans", "Answer"))

for cat, group in grpd:

    most_common = group[' Answer'].value_counts(ascending=False)

    n_ans = most_common.sum()

    if group.shape[0] > 100 or (most_common.iloc[0]/n_ans > 0.001 and n_ans >= 12):

        print("{:>30s}  {:4d}  {:4d}  {:<30s}".format(cat, most_common.shape[0], 

                                                   most_common.iloc[0], most_common.index[0]))
df[' Answer'].value_counts(ascending=False)[:20]
perc_once = answer_counts[answer_counts==1].shape[0]/df.shape[0]

print("Percent of answers only used once: {:.2f}%".format(perc_once*100))
answer_counts = df[' Answer'].value_counts()

fig, ax = plt.subplots()

sns.distplot(answer_counts, ax=ax, kde=False, axlabel="Answer Repeats", norm_hist=True)

plt.yscale('log', nonposy='clip')
categories = df[' Category'].unique()

cats = [x for x in categories if '"' in x]

print("{} word games for {:.2f}% of all categories".format(len(cats), len(cats)/len(categories)*100))