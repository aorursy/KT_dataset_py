%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 500)
train_df =pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
print("Train:", train_df.shape)

print("Test: ", test_df.shape)
train_df.head()
train_df["cat1"].drop_duplicates()
cat_ab = []

cat_not_ab = []

for cat in train_df:

    if cat.startswith("cat"):

        if set(train_df[cat].drop_duplicates()) == set(["A", "B"]):

            cat_ab.append(cat)

        else:

            cat_not_ab.append(cat)

print("cat_ab", str(cat_ab))

print("cat_not_ab", str(cat_not_ab))
reduce(sum, [ord(c) for i, c in enumerate("S")])
from functools import reduce

print(sorted(train_df["cat110"].drop_duplicates(), key=lambda s: reduce(sum, [ord(c) for i, c in enumerate(str(s))])))
cat_lbl = []

for cat in cat_not_ab:

    if cat.startswith("cat"):

        cat_lbl.extend(train_df[cat])

distinct_cat = sorted(set(cat_lbl))

print(distinct_cat)
train_df.dtypes
train_df.describe()
count=1

flag=0

for k in train_df.columns:

    n= sum(train_df[k].isnull())

    if n!=0:    

        print(str(count)+ ': '+str(n)+ '\t',end="")

        flag=1

if flag==0:

    print("No NaN Values")
sns.distplot(train_df['loss'], color = 'r', hist_kws={'alpha': 0.7}, kde = False)
cols= train_df.select_dtypes(include = ['float64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(cols, vmax=1)
cors = cols['loss'].to_dict()

del cors['loss']

for cor in sorted(cors.items(), key = lambda x: -abs(x[1])):

    print("{0}:{1}".

format(*cor))