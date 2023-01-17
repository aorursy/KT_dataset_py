import os, sys, re, time

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path, PosixPath

from IPython.display import HTML, display



CSV_PATH = Path(f'../input/meta-kaggle')

NA_HTML = '<font color=#c0c0c0>?</font>'  # grey '?' means N/A

NROWS = None
plt.rc("figure", figsize=(12,9))

plt.rc("font", size=14)



# Could use DataFrame.style but it would interpret forum message html.

# (ForumMessages.Message Field)

# I want to show that field has raw html...

def df_to_html(df):

    nan_str = '__missing_value_na__'

    html = df.to_html(na_rep=nan_str, notebook=True)

    # to_html escapes html chars, so do a replacement afterwards

    html = html.replace(nan_str, NA_HTML)

    return html



def make_stats(df: pd.DataFrame):

    stats = df.describe(include='all').T

    stats['count'] = stats['count'].astype(int)

    if 'freq' not in stats.columns:

        stats.insert(1, 'freq', np.nan)

    if 'top' not in stats.columns:

        stats.insert(1, 'top', np.nan)

    if 'unique' not in stats.columns:

        stats.insert(1, 'unique', np.nan)

    stats['unique'] = df.nunique()

    stats.insert(0, 'dtype', df.dtypes)

    # add 'top' and 'freq' for numerical columns too

    for c in df.select_dtypes(['number', bool]).columns:

        vc = df[c].value_counts(dropna=False)

        # only show if truly most frequent

        if vc.values[0] > 1:

            stats.loc[c, 'top'] = vc.index[0]

            stats.loc[c, 'freq'] = vc.values[0]

    return stats



# Shows stats and a small sample of a DataFrame loaded from CSV.

# Works on any CSV file, not just Meta Kaggle.

def preview(csv: PosixPath):

    name = csv.with_suffix("").name

    df = pd.read_csv(csv, nrows=NROWS, low_memory=False)

    stats = make_stats(df)

    lines = []

    write = lines.append

    write(f'<h1 id="{name}">{name}</h1>')

    write(f'<h2>Stats</h2>')

    write(f'<p>Rows: {df.shape[0]}')

    write(f'<br/>Columns: {df.shape[1]}')

    write(f'<br/>Memory usage: {df.memory_usage().sum()/(1024**2):,.3f} Mb')

    write(df_to_html(stats))

    write(f'<h2>{name} &mdash; Sample</h2>')

    write(df_to_html(df.sample(n=5, random_state=42).T))

    write(f'<hr/>')

    display(HTML('\n'.join(lines)))
def list_all_ids(csv: PosixPath):

    df = pd.read_csv(csv, nrows=5)

    name = csv.with_suffix('').name

    print()

    for c in df.columns:

        if 'Id' in c:

            print(f'{name} : {c}')
csvs = sorted(CSV_PATH.glob('*.csv'))
!ls -l {CSV_PATH}
for csv in csvs:

    list_all_ids(csv)
for csv in csvs:

    preview(csv)
df = pd.read_csv(CSV_PATH / 'ForumMessageVotes.csv')

df
df.count()
df.nunique()
2298329 / 1149165
df.Id.value_counts()
df.Id.value_counts().value_counts()
df[['Id']].plot(title='ForumMessageVotes duplicates');
df.query('(Id//10)==33333', engine='python').sort_values('Id')
df.drop_duplicates(subset=['Id'])
df.drop_duplicates(subset=['Id']).nunique()
5312782 / 5.74335e+06
1 - 5312782 / 5.74335e+06