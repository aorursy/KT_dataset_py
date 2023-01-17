import numpy as np
import pandas as pd
import seaborn as sns
import os
path = '/kaggle/input/hackathon'
file = f'{path}/task_1-google_search_english_original_metadata.csv'
df = pd.read_csv(file)
f"There are {df.shape[0]} texts originally in English with metadata"
df.head()
df['country'].value_counts()
df['country'].value_counts().hist()
df['is_pdf'].value_counts()
df['language'].value_counts()
df['is_translated'].value_counts()
df['is_downloaded'].value_counts()
df[df['is_downloaded']==False]['char_number'].describe()
df.drop(df[df['is_downloaded']==False].index, inplace=True)
(df['char_number']== 0).mean()
df.drop(df[df['char_number']==0].index, inplace=True)
for _, row in df.sort_values('char_number').head(20).iterrows():
    code = row['alpha_2_code']
    filename=row['filename']
    filename = f'/kaggle/input/hackathon/task_1-google_search_txt_files_v2/{code}/{filename}.txt'
    
    with open(filename, 'r') as file:
        data = file.read()
    print(row['char_number'])
    print(data)
    print('--'*10)
(df['url'].size - df.url.unique().size)/df['url'].size
df['url'].value_counts().head()
path = '/kaggle/input/hackathon'
file = f'{path}/task_1-google_search_translated_to_english_metadata.csv'
df = pd.read_csv(file)
f"There are {df.shape[0]} texts translated to English with metadata"
df.head()
df['country'].value_counts()
df['is_pdf'].value_counts()
df['language'].value_counts()
df['is_translated'].value_counts()
df['is_downloaded'].value_counts()
(df['char_number']== 0).mean()
(df['url'].size - df.url.unique().size)/df['url'].size
df['url'].value_counts().head()
