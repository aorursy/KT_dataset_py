import numpy as np

import pandas as pd 

import seaborn as sns

pd.set_option('max_colwidth', None)
path = '/kaggle/input/hackathon'

files = [f'{path}/task_1-google_search_english_original_metadata.csv',

         f'{path}/task_1-google_search_translated_to_english_metadata.csv']



dfs = []

for file in files:

    df = pd.read_csv(file, encoding = "ISO-8859-1")

    dfs.append(df)

    

df = pd.concat(dfs, ignore_index=True)
f"Considering only {df.shape[0]} sources"
df.head(1)
df.drop(['Is Processed', 'Comments', 'language', 'query'], axis=1, inplace=True)
df[df['alpha_2_code'].isna()].head()
assert all(df[df['alpha_2_code'].isna()]['country']=='Namibia')

df['alpha_2_code'].fillna('NA', inplace=True)
df.drop(df[df['is_downloaded']==False].index, inplace=True)

df['char_number'] = pd.to_numeric(df['char_number'], errors='coerce')

df.drop(df[df['char_number']==0].index, inplace=True)
f"Considering only {df.shape[0]} sources"
df.drop_duplicates('url', keep=False, inplace=True)
f"Considering only {df.shape[0]} sources"
df['char_number'].value_counts().head()
df[df['char_number']==895].head()
df[df['char_number']==895]['url'].str.contains('researchgate.net').mean()
row = df[df['char_number']==895].iloc[0]

code = row['alpha_2_code']

filename=row['filename']

filename = f'/kaggle/input/hackathon/task_1-google_search_txt_files_v2/{code}/{filename}.txt'



with open(filename, 'r') as file:

    data = file.read()



data
df.drop(df[(df['url'].str.contains('researchgate.net')) & (df['char_number']==895)].index, inplace=True)
f"Considering only {df.shape[0]} sources"
from urllib.parse import urlparse

df['url_domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
df['url_domain'].value_counts().head()
df = df[df['url_domain']=='www.ncbi.nlm.nih.gov']
f"Considering only {df.shape[0]} sources"
! pip install pandarallel
import bs4 as bs

import urllib.request

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)



def get_url_title(url):

    try:

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        source = urllib.request.urlopen(req).read()

        soup = bs.BeautifulSoup(source,'lxml')

        if not soup.title:

            print('No title')

            print(url)

            return ""

        return soup.title.text

    except urllib.error.HTTPError as e:

        print(e)

        print(url)

        return ""
df['url_title'] = df['url'].parallel_apply(get_url_title)
df[['country', 'url_title']].head()
df['title_has_country'] = df.apply(lambda row: row['country'] in row['url_title'], axis=1)
df['title_has_country'].value_counts()
df.drop(df[df['title_has_country'] == False].index, inplace=True)
f"Considering only {df.shape[0]} sources"
df['url_title'].value_counts().head()
df[df['url_title']=='The Current Status of BCG Vaccination in Young Children in South Korea']['url']
df.drop_duplicates('url_title', inplace=True)
f"Considering only {df.shape[0]} sources"
df['country'].value_counts()
df['char_number'].plot.box()