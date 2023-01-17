import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def load_metadata(metadata_file):
    df = pd.read_csv(metadata_file,
                 dtype={'Microsoft Academic Paper ID': str,
                        'pubmed_id': str, 
                        "title":str,
                        "absract":str,
                        "WHO #Covidence": str})
    print(f'Loaded metadata with {len(df)} records')
    return df
METADATA_FILE = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

df = load_metadata(METADATA_FILE)
df[["title","abstract"]].head()
def isNaN(string):
    return string != string
print('hello', isNaN('hello'))
print(np.nan,isNaN(np.nan))
df['has_title'] = df.title.apply(lambda x: not isNaN(x) and x != 'TOC')
df['has_abstract'] = df.abstract.apply(lambda x: not isNaN(x) and x != 'TOC')
df['has_title'].value_counts()
df['has_abstract'].value_counts()
n_records = df.shape[0]
pct_has_title = df['has_title'].sum()/n_records * 100
pct_has_abstract = df['has_abstract'].sum()/n_records * 100
pct_has_title_and_abstract = df[df['has_title'] & df['has_abstract']].shape[0]/n_records * 100
pct_has_title_or_abstract = df[df['has_title'] | df['has_abstract']].shape[0]/n_records * 100

print(f"Number of records: {n_records}")
print(f"Records with title: {pct_has_title:.2f}%")
print(f"Records with abstract: {pct_has_abstract:.2f}%")
print(f"Records with both: {pct_has_title_and_abstract:.2f}%")
print(f"Records with text: {pct_has_title_or_abstract:.2f}%")
df['title_len'] = df["title"].apply(lambda x : len(str(x)))
df['abstract_len'] = df["abstract"].apply(lambda x : len(str(x)))
df[df.has_title].title_len.hist(bins=30)
df[df.has_abstract].abstract_len.hist()
df[df.title_len < 15][["title_len","title","abstract"]]
df[~df.has_abstract][["title_len","abstract_len","title","abstract"]]
df[(df.abstract_len > 5) & (df.abstract_len < 30) ][["title_len","abstract_len","title","abstract"]]
import langdetect as ld

SEED= 53 

from langdetect import DetectorFactory
DetectorFactory.seed = SEED
def lang_detect(title, abstract):
    try:
        str_abstract = '' if (isNaN(abstract)) else abstract
        str_title = '' if (isNaN(title)) else title
        return ld.detect((str_title + ' ' + str_abstract).strip())
    except: 
        return '--'

print(lang_detect(df.iloc[0].title, 'Hola'))
print(lang_detect(df.iloc[0].title, df.iloc[0].abstract))
print(lang_detect(df.iloc[0].title, np.nan))
print(lang_detect(np.nan, df.iloc[0].title))
print(lang_detect(np.nan, np.nan))
df.head().title
df.head().apply(lambda x: lang_detect(x.title, x.abstract), axis = 1)
df['lang'] = df.apply(lambda x: lang_detect(x.title, x.abstract), axis = 1)
df['lang'].value_counts().plot(kind="barh")
df['lang'].value_counts()
df['lang'].value_counts()/n_records
show_cols = ['lang','source_x','title','abstract','publish_time','journal','WHO #Covidence']
df[df['lang']=='en'][show_cols].head()
df[df['lang']=='fr'][show_cols]
df[df['lang']=='es'][show_cols]
df[df['lang']=='de'][show_cols]
df[df['lang']=='it'][show_cols]
df[df['lang']=='zh-cn'][show_cols]
df['lang'].apply(lambda x: x not in ['en','fr','es','de'])
df[df['lang'].apply(lambda x: x not in ['en','fr','es','de','it'])][show_cols]
