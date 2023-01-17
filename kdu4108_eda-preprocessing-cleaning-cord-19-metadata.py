# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggl/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls /kaggle/input/CORD-19-research-challenge/
root_path = '/kaggle/input/CORD-19-research-challenge/'

metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})

meta_df.head()
meta_df.info()
def find_duplicates(df, cols):

    dups = df.duplicated(subset=cols, keep='first')

    dups_counter_dict = {"title": Counter(),

                         "doi": Counter(), 

                         "abstract": Counter()}

    for i in range(len(dups[dups].index)):

        article = df.loc[dups[dups].index[i]]

        dups_counter_dict["title"][article.title] += 1

        dups_counter_dict["doi"][article.doi] += 1

        dups_counter_dict["abstract"][article.abstract] += 1



    return dups, dups_counter_dict, dups.sum()
dup_titles, dup_titles_counter, num_dup_titles = find_duplicates(meta_df, ["title"])

most_common_titles = dup_titles_counter['title'].most_common()

print(f"Number of duplicates: {num_dup_titles}")

print(f"Duplicate titles: {dup_titles_counter['title'].most_common()}")
print(meta_df[meta_df.title == "Fractional Dosing of Yellow Fever Vaccine to Extend Supply: A Modeling Study"]) # Need to drop
print(meta_df[meta_df.title == "Middle East respiratory syndrome"]) # Don't need to drop any.
for (title, count) in most_common_titles:

    if count <= 10:

        break

    else:

        meta_df = meta_df[~(meta_df.title == title)]

        
meta_df.info()
dup_doi, dup_doi_counter, num_dup_dois = find_duplicates(meta_df, ["doi"])

print(f"Number of duplicates: {num_dup_dois}")

print(f"Duplicate titles: {dup_doi_counter['title'].most_common(20)}")

print(f"Duplicate dois: {dup_doi_counter['doi'].most_common()}")
not_null_doi = meta_df[meta_df['doi'].notnull()]

# not_null_doi.info()

dup_doi_nonnull, dup_doi_nonnull_counter, num_dup_doi_nonnull = find_duplicates(not_null_doi, ["doi"])

print(f"Number of duplicates: {num_dup_doi_nonnull}")

print(f"Duplicate titles: {dup_doi_nonnull_counter['title'].most_common()}")

# returns empty set, suggesting there are 0 duplicate doi's that are not null
dup_abstract, dup_abstract_counter, num_dup_abstract = find_duplicates(meta_df, ["abstract"])

print(f"Number of duplicates: {num_dup_abstract}")

print(f"Duplicate titles: {dup_abstract_counter['title'].most_common(20)}")

print(f"Duplicate abstracts: {dup_abstract_counter['abstract'].most_common(20)}")
dup_abstract[dup_abstract].index

print(f"Number of null abstracts: {meta_df.abstract.isnull().sum()}")

print(f"First duplicate abstract: {meta_df.loc[dup_abstract[dup_abstract].index[0]].abstract}")
not_null_abstract = meta_df[meta_df['abstract'].notnull()]

dup_abstract_nonnull, dup_abstract_nonnull_counter, num_dup_abstract_nonnull = find_duplicates(not_null_abstract, ["abstract"])

print(f"Number of duplicates: {num_dup_abstract_nonnull}")

print(f"Duplicate titles: {dup_abstract_nonnull_counter['title'].most_common(20)}")

print(f"Duplicate abstracts: {dup_abstract_nonnull_counter['abstract'].most_common(20)}")
meta_df.replace('Unknown', np.NaN, inplace=True)

print(not_null_abstract[not_null_abstract.title == "The Heritage of Pathogen Pressures and Ancient Demography in the Human Innate-Immunity CD209/CD209L Region"])
print(not_null_abstract[not_null_abstract.title == "Fractional Dosing of Yellow Fever Vaccine to Extend Supply: A Modeling Study"])
abstract_nan_df = meta_df[pd.isnull(meta_df.abstract)]

# abstract_nan_df.info()

# abstract_nan_df.head(20)

sources = abstract_nan_df.source_x.unique()

print("Sources:", sources)



for source in sources:

    print("Current source:", source)

    # Checking Elsevier

    print(abstract_nan_df[abstract_nan_df.source_x == source].info()) # 6634 total articles

    print(abstract_nan_df[abstract_nan_df.source_x == source].sample(5)) # seems none have abstracts?

# meta_df_condensed_abs = meta_df.fillna("").groupby(['abstract']).max().reset_index() # TODO: make this work with fuzzy matching
meta_df_cond_abs_nonnull = meta_df[meta_df["abstract"].notnull()].fillna("").groupby(['abstract']).max().reset_index() 

meta_df_abs_null = meta_df[meta_df["abstract"].isnull()]

meta_df_cond_abs = pd.concat([meta_df_cond_abs_nonnull, meta_df_abs_null])
print(meta_df.info())

print(meta_df_cond_abs.info())

print(meta_df_cond_abs[meta_df_cond_abs.title == "The Heritage of Pathogen Pressures and Ancient Demography in the Human Innate-Immunity CD209/CD209L Region"])
print(meta_df_cond_abs[meta_df_cond_abs.title == "Fractional Dosing of Yellow Fever Vaccine to Extend Supply: A Modeling Study"])
def doi_url(d):

    if d.startswith('http://'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        url = f'http://doi.org/{d}'

        if url == "http://doi.org/":

            return np.nan

        return url

meta_df_cond_abs.doi = meta_df_cond_abs.doi.fillna('').apply(doi_url)

meta_df_cond_abs.to_csv("cleaned_metadata.csv")