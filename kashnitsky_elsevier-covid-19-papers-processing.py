import json

from pathlib import Path

from tqdm.notebook import tqdm

import pandas as pd

from pprint import pprint
fname = '../input/covid19-articles-by-elsevier/meta/meta/0022510X79902181_meta.json'
with open(fname) as f:

    json_data = json.load(f)
json_data['full-text-retrieval-response']['coredata'].keys()
json_data['full-text-retrieval-response']['coredata']['dc:title']
json_data['full-text-retrieval-response']['coredata']['dc:description']
pprint(json_data['full-text-retrieval-response'])
json_data['full-text-retrieval-response']['coredata']['link']
def process_abstract(text):

    return " ".join(text.replace('\n', '').replace('\t', '').replace('Abstract', '').strip().split())
process_abstract(json_data['full-text-retrieval-response']['coredata']['dc:description'])
def extract_fields_from_json(fname):

    with open(fname) as f:

        json_data = json.load(f)

        

    core_data = json_data['full-text-retrieval-response']['coredata']

    

    title = core_data.get('dc:title', None)

    abstract = core_data.get('dc:description', None)

    doi = core_data.get('prism:doi', None)

    date = core_data.get('prism:coverDate', None)   

    xml_url = core_data.get('prism:url', None)

    url = core_data['link'][1]['@href']

    

    proc_abstract = None

    if abstract:

        proc_abstract = process_abstract(abstract)

    

    return title, proc_abstract, doi, date, xml_url, url
def jsons_to_dataframe(path_to_data='meta',

                       file_mask = '*.json',

                       min_abstract_len_words=40,

                       columns=('title', 'abstract', 'doi', 'date', 'xml_url', 'url')):

    

    all_data = []

    for fname in tqdm(Path(path_to_data).glob(file_mask)):

        try:

            title, proc_abstract, doi, date, xml_url, url = extract_fields_from_json(fname)

        except json.JSONDecodeError:

            continue



        if title and proc_abstract and len(proc_abstract.split()) >= min_abstract_len_words:

            all_data.append([title, proc_abstract, doi, date, xml_url, url])

            

    df = pd.DataFrame(all_data, columns=columns)

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by='date').reset_index(drop=True)

    

    return df   
df = jsons_to_dataframe(path_to_data='../input/covid19-articles-by-elsevier/meta/meta/')
df.head()
df.info()
df.tail()
train_df, valid_df = df.head(10000), df.tail(2967).reset_index(drop=True)
train_df.to_csv('covid_artilces_elsevier_train.csv')

valid_df.to_csv('covid_artilces_elsevier_validation.csv')