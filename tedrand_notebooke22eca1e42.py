import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline
df_docmet = pd.read_csv('../input/documents_meta.csv')

df_docent = pd.read_csv('../input/documents_entities.csv')

df_doccat = pd.read_csv('../input/documents_categories.csv')

df_doctop = pd.read_csv('../input/documents_topics.csv')

df_procon = pd.read_csv('../input/promoted_content.csv')

df_events = pd.read_csv('../input/events.csv')

df_viewsamp = pd.read_csv('../input/page_views_sample.csv')
df_docmet.head()
df_docmet.publisher_id.unique()
df_docent.head()
entity_conf_hist = plt.hist(df_docent.confidence_level, bins=20)
df_doccat.head()
category_conf_hist = plt.hist(df_doccat.confidence_level, bins=20)
df_viewsamp.head()