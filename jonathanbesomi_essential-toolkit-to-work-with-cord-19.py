# Install and import texthero

!pip install texthero -q

import texthero as hero



# Import the other packages

import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt



import seaborn as sns

sns.set(color_codes=True)



from pathlib import Path

import glob

import json
# Concat



def get_data(metadata_only=False):

    """

    Return CORD-19 dataset

    

    Parameters

    ----------

    

    metadata_only : bool (False by default )

        - When True, returns only the metadata Pandas DataFrame

    

    """

    if metadata_only:

        return pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

    

    CLEAN_DATA_PATH = Path("../input/cord-19-eda-parse-json-and-generate-clean-csv/")



    biorxiv_df = pd.read_csv(CLEAN_DATA_PATH / "biorxiv_clean.csv")

    biorxiv_df['source'] = 'biorxiv'



    pmc_df = pd.read_csv(CLEAN_DATA_PATH / "clean_pmc.csv")

    pmc_df['source'] = 'pmc'



    comm_use_df = pd.read_csv(CLEAN_DATA_PATH / "clean_comm_use.csv")

    comm_use_df['source'] = 'comm_use'



    noncomm_use_df = pd.read_csv(CLEAN_DATA_PATH / "clean_noncomm_use.csv")

    noncomm_use_df['source'] = 'noncomm_use'



    papers_df = pd.concat(

        [biorxiv_df,pmc_df, comm_use_df, noncomm_use_df], axis=0

    ).reset_index(drop=True)



    

    return papers_df
papers_df = get_data()

papers_df.head()
papers_df.shape
def tfidf(df, columns=['text', 'abstract'], dim=256):

        

    if len(columns) == 0:

        raise ValueError("columns argument must be a least and have at least one value.")

    

    # Merge all text columns

    df['content'] = df[columns[0]]

    

    for col in columns[1:]:

        df['content'] += df[col]

        

    # Fill missing NA

    if df['content'].isna().sum() > 0:

        print("Warning. The dataset contains NA. They will be dropped for TF-IDF computation.")

        content = df['content'].dropna()

    else:

        content = df['content']

    

    # Compute TF-IDF

    return content.pipe(hero.do_tfidf, max_features=dim)
sample_df = papers_df.sample(1000)

tfidf_s = tfidf(sample_df, columns=['abstract'])

tfidf_s.head()
sample_df = papers_df.sample(5000)

sample_df['tfidf'] = tfidf(sample_df, columns=['abstract'])

sample_df.head(2)
sample_df['tfidf'].isna().sum()
def pca(s):

    return hero.do_pca(s)
sample_df = sample_df.dropna(how='any')

sample_df['pca'] = sample_df['tfidf'].pipe(pca)

sample_df['pca'].head()
def show_pca(df, pca_col, color_col=None, title=""):

    return hero.scatterplot(df, pca_col, color=color_col, title=title)
title = "Vector space representation of CORD-19"

show_pca(sample_df, 'pca', title=title)
def kmeans(s, n_clusters):

    return hero.do_kmeans(s, n_clusters=n_clusters)
title = "Vector space representation of CORD-19 with K-means"

sample_df['kmeans'] = kmeans(sample_df['tfidf'], 20)

sample_df['kmeans'] = sample_df['kmeans'].astype(str)  # for a nicer visualization

show_pca(sample_df, 'pca', color_col='kmeans', title=title)
papers_df.to_csv("cord19.csv", index=False)

get_data(metadata_only=True).to_csv("cord19_metadata_only.csv", index=False)