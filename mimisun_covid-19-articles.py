import pandas as pd

print(f"Last updated on {pd.to_datetime('today').strftime('%m/%d/%Y')}")
import glob

import json

import os

import numpy as np

import pandas as pd

from pprint import pprint



input_path = '/kaggle/input/CORD-19-research-challenge/'



# See JSON schema here: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge#json_schema.txt

def read_text(file, key):

    entries = file[key]

    if isinstance(entries, dict):

        entries = entries.values()

    return '\n'.join([x['text'] for x in entries])



# Each paragraph is a separate entry in the body_text data.

def read_body_text(file):

    prev_section = None

    text = []

    for x in file['body_text']:

        section = x['section']

        if section != prev_section:

            text.append('<SECTION>')

            text.append(section)

            prev_section = section

        text.append(x['text'])

    return '\n'.join(text)

    

def read_file(filename):

    file = json.load(open(filename, 'rb'))

    data = {

        'paper_id': file['paper_id'],

        'title': file['metadata']['title'],

        'abstract': read_text(file, 'abstract'),

        'body_text': read_body_text(file),

        'ref_entries': read_text(file, 'ref_entries'),

    }

    return data
json_files = sorted(glob.glob(f'{input_path}/**/pdf_json/*.json', recursive=True))

print('Total pdf files: ', len(json_files))



file_contents = []

for i, file in enumerate(json_files):

    if i % 2000 == 0:

        print(f'Processing {i} of {len(json_files)}')

    file_contents.append(read_file(file))

content_df = pd.DataFrame(file_contents)
metadata_path = f'{input_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'sha': str,

    'doi': str,

    'pmcid': str,

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str,

    'WHO #Covidence': str,

})

pd.options.display.max_colwidth = 100

meta_df.info()
print('PDF-only: ', meta_df[meta_df.has_pdf_parse & ~meta_df.has_pmc_xml_parse].cord_uid.count())

print('PMC-only: ', meta_df[~meta_df.has_pdf_parse & meta_df.has_pmc_xml_parse].cord_uid.count())
full_df = content_df.merge(meta_df, how='inner', left_on='paper_id', right_on='sha')

full_df = full_df.replace(r'^\s*$', np.nan, regex=True)

full_df.info()
full_df['title'] = full_df.title_y.fillna(full_df.title_x)

full_df['abstract'] = full_df.abstract_y.fillna(full_df.abstract_x)

full_df.drop(columns=['sha', 'title_x', 'title_y', 'abstract_x', 'abstract_y'], inplace=True)

full_df.info()
stats = {}

stats['total'] = meta_df.shape[0]

stats['full text'] = full_df.shape[0]

df = full_df



# Drop duplicates by title (an article may come from multiple sources resulting in dups).

df = df.drop_duplicates(['title'])

stats['uniques'] = df.shape[0]

df.to_csv('cord19.csv', index=False)



# Filter by publish time.

df = df[pd.to_datetime(df.publish_time, errors='coerce') >= pd.to_datetime('2019-12-01')]

stats['after 12/01/2019'] = df.shape[0]



# Filter by covid-19 keywords.

keywords = ['novel coronavirus', 'COVID-19', 'COVID19', 'SARS-CoV-2', '2019-nCov']

keywords_regex = '|'.join(keywords)

df_covid19 = df[df.title.str.contains(keywords_regex, na=False, case=False, regex=True) | df.abstract.str.contains(keywords_regex, na=False, case=False, regex=True)]

stats['covid-19'] = df_covid19.shape[0] 

df_covid19.to_csv('covid19.csv', index=False)
# Write full metadata as well:

meta_df.to_csv('cord19_meta.csv', index=False)
# Also write covid19_metadata for all articles, including those without full text.

df = meta_df

df = df.drop_duplicates(['title'])

df = df[pd.to_datetime(df.publish_time, errors='coerce') >= pd.to_datetime('2019-12-01')]

df_covid19_meta = df[df.title.str.contains(keywords_regex, na=False, case=False, regex=True) | df.abstract.str.contains(keywords_regex, na=False, case=False, regex=True)]

stats['covid-19-meta'] = df_covid19_meta.shape[0] 

df_covid19_meta.to_csv('covid19_meta.csv', index=False)

pd.DataFrame([stats]).transpose()
df_covid19_meta.head(1).transpose()
df_covid19.head(1).transpose()
print(df_covid19.iloc[0].abstract)
print(df_covid19.iloc[0].body_text.split('<SECTION>')[1])
meta_df[meta_df.title.str.contains('Coronaviruses: a paradigm of new emerging zoonotic diseases', case=False, na=False)]