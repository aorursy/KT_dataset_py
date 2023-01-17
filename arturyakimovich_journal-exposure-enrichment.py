import pandas as pd

# load covid metadata table


metadata_path = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

metadata = pd.read_csv(metadata_path)
metadata.head()
df = metadata
df = df[df['authors'].str.contains("Yakimovich", na=False)]
df.head()
#load scimago path
import os

scimago_path = os.path.join('/kaggle/input','scimagojournalcountryrank','scimagojr 2018.csv')

scimago_data = pd.read_csv(scimago_path, sep=';')
scimago_data.head()
# join journals
metadata_preprints = metadata[(metadata['source_x']=='biorxiv')|(metadata['source_x']=='medrxiv')]
metadata_journals =  metadata.dropna(subset=['journal'])

enriched_metadata = pd.merge(left=metadata_journals, right=scimago_data, left_on='journal', right_on='Title')
print('total metadata entries: {} (of which prerints {} and journals {}), scimago entries: {}, matching entries: {}'.format(len(metadata),len(metadata_preprints), len(metadata_journals), len(scimago_data), len(enriched_metadata)))
enriched_metadata.head()
# let's explore the distribution of H-factors in the Kaggle literature
import matplotlib.pyplot as plt
import seaborn as sns


ax = sns.distplot(enriched_metadata['H index'].values)
enriched_metadata[enriched_metadata['H index']>1000].head()
enriched_metadata['exposure'] = pd.cut(enriched_metadata['H index'], bins=[0, 10, 50, 10000], labels=["low", "medium", "high"])

enriched_metadata.to_csv('/kaggle/working/enriched_metadata.csv')
enriched_metadata[enriched_metadata['exposure']=='low'].head()