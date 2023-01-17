# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

color = sns.color_palette()

%matplotlib inline

answers = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
def parseSize(size):

    if size.endswith("MB"):

        if size.startswith("<"):

            size = int(size.rstrip("MB").lstrip("<"))

        else:

            size = int(size.rstrip("MB")) * 10**3

    elif size.endswith("GB"):

        size = float(size.rstrip("GB")) * 10**6

    elif size.endswith("TB"):

        size = float(size.rstrip("TB")) * 10**9

    elif size.endswith("PB"):

        size = float(size.rstrip("PB")) * 10**12

    elif size.endswith("EB"):

        if size.startswith(">"):

            size = int(size.rstrip("EB").lstrip(">")) * 10**16

        else:

            size = int(size.rstrip("EB")) * 10**15  

    return int(size)

df = pd.DataFrame(data=answers.WorkDatasetSize.dropna())

counts = df.WorkDatasetSize.value_counts()

counts
df = pd.DataFrame(df['WorkDatasetSize'].apply(parseSize))
result = df.sort_values(['WorkDatasetSize'])
data_sizes = result['WorkDatasetSize']

dataset_count = data_sizes.value_counts()
df_counts = pd.DataFrame(dataset_count)

df_counts = df_counts.sort_index()
df_counts.index = df_counts.index / 1000000

df_counts.plot(kind="bar", title="Dataset Size (GB)", legend=False)