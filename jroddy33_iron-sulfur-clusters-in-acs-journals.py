import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_bc = pd.read_json('/kaggle/input/american-chemical-society-journals/biochem.json')

df_jacs = pd.read_json('/kaggle/input/american-chemical-society-journals/jacs.json')

df_ic = pd.read_json('/kaggle/input/american-chemical-society-journals/inorg_chem.json')

df_om = pd.read_json('/kaggle/input/american-chemical-society-journals/organometallics.json')

df_cr = pd.read_json('/kaggle/input/american-chemical-society-journals/chem_revs.json')



df = pd.concat([df_bc, df_jacs, df_ic, df_om, df_cr])

print(df.describe())

print(df.columns)
plt.style.use('ggplot')

df.groupby(['journal', 'year']).year.count().unstack('journal').plot()

df.groupby('year').year.count().plot(label='All 5')

plt.ylabel('Articles')

plt.legend()

plt.title('Articles by Journal')



journal_grouped = df.groupby('journal').journal.count()

print(journal_grouped)

print('total:', journal_grouped.sum())
print(df.groupby('article_type').article_type.count())
print("Length before cleaning:", len(df))



unwanted_article_types = ['Book Review', 'Computer Software Review', 'Announcement', 'Spotlights', "Editor's Page", 'Roundtable']

uat_mask = df.article_type.apply(lambda x: x in unwanted_article_types)

df_clean = df[~uat_mask]

print("Length after removing article types:", len(df_clean))



unwanted_titles = df_clean.groupby('title').title.count()

unwanted_titles = list(unwanted_titles[unwanted_titles>3].index)

ut_mask = df_clean.title.apply(lambda x: x in unwanted_titles)

df_clean = df_clean[~ut_mask]

print("Length after removing article titles:", len(df_clean))

dlms_mask = df_clean.authors.apply(lambda x: 'Daniel L. M. Suess' in x)

df_clean[dlms_mask]
fen_sn = re.compile('Fe[1-9]-?Se?[1-9]') # iron sulfur/selenium clusters

prot_fen_sn = re.compile('[1-9]Fe-?[1-9]Se?') # iron sulfur/selenium clusters (protein notation)

con_sn = re.compile('Co[1-9]-?Se?[1-9]') # cobalt sulfur/selenium clusters

femoco = re.compile('(?i)femoco') # FeMoco

nitrogenase = re.compile('(?i)nitrogenase') # nitrogenase



regexes = [fen_sn, prot_fen_sn, con_sn, femoco, nitrogenase,]



masks = []

for r in regexes:

    mask = df_clean.title.apply(lambda x: bool(r.search(x))) | df_clean.abstract.apply(lambda x: bool(r.search(x)))

    masks.append(mask)

    

masks = np.array(masks).T

mask = masks.any(axis=1)
plt.style.use('ggplot')

df_clean[mask].groupby('year').year.count().plot()

plt.ylabel('Articles')

plt.title('Prevalence of Articles Related to Iron-Sulfur Clusters')