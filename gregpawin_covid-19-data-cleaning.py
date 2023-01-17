!pip install spacy
!pip install spacy_langdetect
!pip install matplotlib
!pip install numpy
!pip install pandas
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import spacy
from spacy_langdetect import LanguageDetector
import re
cdc = pd.read_excel('../input/cdc-covid19-research-articles/All_Articles_Excel.xlsx', parse_dates=True)
# Examining all the unnamed columns
cdc.iloc[:,16:].dropna(how='all')
# Mostly nulls, html tags, and foreign language data, so drop
cdc = cdc.iloc[:,:16]
cdc.columns
# Plot the dates for all entries
register_matplotlib_converters()
plt.figure(figsize=(15,8))
plt.hist(cdc['Date Added'], bins=20)
plt.title('Histogram of Number of Articles Published by Date')
plt.xticks(rotation=90)
plt.show()
# Abstracts with integer data types
# Columns from Title to Keywords are erronously shifted
cdc[cdc['Abstract'].apply(lambda x: isinstance(x,int))]
# Shifted data
shifted_columns = ['Abstract', 'Year',
       'Journal/Publisher', 'Volume', 'Issue', 'Pages', 'Accession Number',
       'DOI', 'URL', 'Name of Database', 'Database Provider', 'Language',
       'Keywords']
shifted_data = cdc[cdc['Abstract'].apply(lambda x: isinstance(x,int))].shift(1, axis=1)[shifted_columns]
# Correcting shifted data
cdc.loc[cdc['Abstract'].apply(lambda x: isinstance(x,int)), shifted_columns] = shifted_data
cord = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', parse_dates=True, low_memory=False)
cord.info()
cord[cord.title.isna()].abstract
# Remove records with missing titles
cord = cord[cord.title.notna()]
len(cord[cord.abstract.isna()])
import en_core_sci_lg
nlp = en_core_sci_lg.load()
nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
def lang_detect_en(text):
    doc = nlp(text)
    if doc._.language.get('language') == 'en':
        return doc._.language.get('score')
    else:
        return 0
cord['lang_detect'] = cord['title'].apply(lang_detect_en)
# Remove records with missing titles
cdc = cdc[cdc.Title.notna()]
cdc['lang_detect'] = cdc['Title'].apply(lang_detect_en)
cord = cord[cord['lang_detect'] != 0]
cdc = cdc[cdc['lang_detect'] != 0]
interested_columns = ['title','authors','abstract','publish_time','url']
interested_columns_cdc = ['Title','Author','Abstract','Year','URL']
cord_subset = cord[interested_columns].copy()
cdc_subset = cdc[interested_columns_cdc].copy()
cdc_subset.columns = interested_columns
combined_dataset = cord_subset.append(cdc_subset).reset_index(drop=True)
len(combined_dataset)
combined_dataset['combined_text'] = combined_dataset.title + str(combined_dataset.abstract)
combined_dataset['combined_text'] = combined_dataset['combined_text'].str.lower()
combined_dataset['combined_text'] = combined_dataset.combined_text.apply(lambda x: re.sub(r'[\W_]+', ' ', x))
for text in combined_dataset.loc[combined_dataset.duplicated(subset=['combined_text']), 'combined_text']:
  dup_list = combined_dataset.loc[combined_dataset.combined_text == text,'url']
  #print(len(dup_list), len(dup_list.notna()), len(dup_list.isna()))
  if dup_list.notna().sum() == 0:
    combined_dataset.drop(dup_list[1:].index, inplace=True)
  else:
    combined_dataset.drop(dup_list[dup_list.notna()][1:].index, inplace=True)
    combined_dataset.drop(dup_list[dup_list.isna()].index, inplace=True)
combined_dataset = combined_dataset.reset_index(drop=True)
    
len(combined_dataset)
combined_dataset = combined_dataset[combined_dataset.abstract.notna()].reset_index(drop=True)
len(combined_dataset)
combined_dataset.to_csv('combined_dataset.csv')