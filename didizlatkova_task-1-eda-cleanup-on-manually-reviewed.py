import numpy as np

import pandas as pd

import seaborn as sns

import os
path = '/kaggle/input/hackathon'

file = f'{path}/task_1-google_search_manually_reviewed_metadata.csv'

df = pd.read_csv(file, encoding = "ISO-8859-1")
f"There are {df.shape[0]} manually reviewed texts with metadata"
df.head()
df.info()
df[df['filename'].isnull()]
df.drop(df[df['filename'].isna()].index, inplace=True)
df['language'].value_counts()
df['is_pdf'].value_counts()
df['is_translated'].value_counts() # All original
df['is_downloaded'].value_counts() # All downloaded
df['Is Processed'].value_counts()
df['country'].unique().size
pd.set_option('display.max_colwidth', None)
df[df['Comments'].notna()]['Comments']
df['Snippet'].isna().mean()
df.drop(df[df['Snippet'].isna()].index, inplace=True)
df['snippet_len'] = df['Snippet'].astype(str).apply(len)
df['snippet_len'].hist()
def is_snippet_in_text(row):

    code = row['alpha_2_code']

    file = row['filename']

    if not file.endswith('.txt'):

        file += '.txt'

    filename=f'{path}/task_1-google_search_txt_files_v2/{code}/{file}'

    if os.path.isfile(filename):

        with open(filename, 'r') as file:

            data = file.read()

        return row['Snippet'] in data
df['snippet_in_text'] = df.apply(is_snippet_in_text, axis=1)
df['snippet_in_text'].mean()
def get_content_len(row):

    code = row['alpha_2_code']

    file = row['filename']

    if not file.endswith('.txt'):

        file += '.txt'

    filename=f'{path}/task_1-google_search_txt_files_v2/{code}/{file}'

    if os.path.isfile(filename):

        with open(filename, 'r') as file:

            data = file.read()

        return len(data)

    else:

        print(f"Could not find file {filename} in folder {code}")
df['text_len'] = df.apply(get_content_len, axis=1)
df[df['text_len'].isna()]
df.drop(df[df['text_len'].isna()].index, inplace=True)
df.shape
df.drop(['query','language','is_translated','is_downloaded','char_number','Is Processed'], inplace=True, axis=1)
df.to_csv('/kaggle/working/manually_reviewed_cleaned.csv', index=False)