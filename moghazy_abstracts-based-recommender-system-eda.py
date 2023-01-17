import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import os
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

print("Cols names: {}".format(meta.columns))

meta.head(7)
plt.figure(figsize=(20,10))

meta.isna().sum().plot(kind='bar', stacked=True)
meta_dropped = meta.drop(['Microsoft Academic Paper ID', 'WHO #Covidence'], axis = 1)
plt.figure(figsize=(20,10))



meta_dropped.isna().sum().plot(kind='bar', stacked=True)
miss = meta['abstract'].isna().sum()

print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))
abstracts_papers = meta[meta['abstract'].notna()]

print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))

missing_doi = abstracts_papers['doi'].isna().sum()

print("The number of papers without doi is {:0.0f}".format(missing_doi))

missing_url = abstracts_papers['url'].isna().sum()

print("The number of papers without url is {:0.0f}".format(missing_url))
missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]

print("The total number of papers with abstracts but missing url and missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))