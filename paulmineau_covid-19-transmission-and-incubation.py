import covid19_tools as cv19

import pandas as pd

import re

from IPython.core.display import display, HTML



METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'



# Load metadata

meta = cv19.load_metadata(METADATA_FILE)

# Add tags

meta, covid19_counts = cv19.add_tag_covid19(meta)

meta, transmission_counts = cv19.count_and_tag(meta,

                                               cv19.TRANSMISSION_SYNONYMS,

                                               'transmission_generic')

meta, repr_counts = cv19.count_and_tag(meta,

                                       cv19.REPR_SYNONYMS,

                                       'transmission_repr')

meta, incubation_counts = cv19.count_and_tag(meta,

                                             cv19.INCUBATION_SYNONYMS,

                                             'transmission_incub')

meta, persistence_counts = cv19.count_and_tag(meta,

                                              cv19.PERSISTENCE_SYNONYMS,

                                              'transmission_persist')



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
print('Loading full text for tag_disease_covid19')

full_text_repr = cv19.load_full_text(meta[meta.tag_disease_covid19 &

                                          meta.tag_transmission_repr],

                                     '../input/CORD-19-research-challenge')
table_repr = []

repr_strings = [r'\br0\b', r'$r_0$', r'\br 0\b',

                'reproduction number', 'reproduction rate',

                'reproductive number', 'rate of reproduction']



for i, record in enumerate(full_text_repr):

    sha = record['paper_id']

    meta_row = meta[meta.sha == sha]

    temp_dict = {

        'publication_date': meta_row.publish_time.values[0],

        'authors': meta_row.authors_short.values[0],

        'doi': meta_row.doi.values[0],

        'title': meta_row.title.values[0],

        'journal': meta_row.journal.values[0],

        'key_passages': []

    }



    for item in record['body_text']:

        if 'value of' in item['text'].lower() or 'estimated' in item['text'].lower():

            sentences = item['text'].split('. ')

            for s in sentences:

                if len(re.findall('|'.join(repr_strings), s.lower())) > 0:

                    if len(re.findall(r'\d+\.\d+', s)) > 0:

                        temp_dict['key_passages'].append(s)

    if len(temp_dict['key_passages']) == 0:

        temp_dict['key_passages'] = ['<i>Failed to extract figures - check manually.</i>']

    table_repr.append(temp_dict)

    

table_repr = pd.DataFrame(table_repr)



table_repr['key_passages'] = (table_repr

                              .key_passages

                              .apply(lambda x: '<br><br>'.join(x)))

table_repr['title_link'] = table_repr.apply(lambda x: f'<a href="{x.doi}">{x.title}</a> ({x.journal})',

                                                 axis=1)

# table_repr.drop(['title', 'doi'], axis=1, inplace=True)
cv19.display_dataframe(table_repr[['publication_date',

                                   'authors',

                                   'title_link',

                                   'key_passages']],

                       'Table of Reproduction Rates (<i>R</i> / <i>R<sub>0</sub></i>)')
print('Loading full text for tag_transmission_incub')

full_text_incub = cv19.load_full_text(meta[meta.tag_disease_covid19 &

                                           meta.tag_transmission_incub],

                                      '../input/CORD-19-research-challenge')
table_incub = []

time_strings = []

for unit in ['day', 'hour', 'hr']:

    time_strings += [f'\\d+ {unit}', f'\\d+\\.\\d+ {unit}', f'{unit} \\d']



for i, record in enumerate(full_text_incub):

    sha = record['paper_id']

    meta_row = meta[meta.sha == sha]

    temp_dict = {

        'publication_date': meta_row.publish_time.values[0],

        'authors': meta_row.authors_short.values[0],

        'doi': meta_row.doi.values[0],

        'title': meta_row.title.values[0],

        'journal': meta_row.journal.values[0],

        'key_passages': []

    }



    for item in record['body_text']:

        if 'value of' in item['text'].lower() or 'estimated' in item['text'].lower():

            sentences = item['text'].split('. ')

            for s in sentences:

                if len(re.findall('|'.join(cv19.INCUBATION_SYNONYMS), s.lower())) > 0:

                    if len(re.findall('|'.join(time_strings), s.lower())) > 0:

                        temp_dict['key_passages'].append(s)

    if len(temp_dict['key_passages']) == 0:

        temp_dict['key_passages'] = ['<i>Failed to extract figures - check manually.</i>']

    table_incub.append(temp_dict)

    

table_incub = pd.DataFrame(table_incub)



table_incub['key_passages'] = (table_incub

                                 .key_passages

                                 .apply(lambda x: '<br><br>'.join(x)))

table_incub['title_link'] = table_incub.apply(lambda x: f'<a href="{x.doi}">{x.title}</a> ({x.journal})',

                                                    axis=1)

# table_incub.drop(['title', 'doi'], axis=1, inplace=True)
cv19.display_dataframe(table_incub[['publication_date',

                                    'authors',

                                    'title_link',

                                    'key_passages']],

                       'Table of Incubation Periods')
print('Loading full text for tag_transmission_persist')

full_text_persist = cv19.load_full_text(meta[meta.tag_disease_covid19 &

                                             meta.tag_transmission_persist],

                                             '../input/CORD-19-research-challenge')
table_persist = []

time_strings = []

for unit in ['day', 'minute', 'min', 'hour', 'hr', 'second', 'sec']:

    time_strings += [f'\\d+ {unit}', f'\\d+\\.\\d+ {unit}', f'{unit} \\d']



for i, record in enumerate(full_text_persist):

    sha = record['paper_id']

    meta_row = meta[meta.sha == sha]

    temp_dict = {

        'publication_date': meta_row.publish_time.values[0],

        'authors': meta_row.authors_short.values[0],

        'doi': meta_row.doi.values[0],

        'title': meta_row.title.values[0],

        'journal': meta_row.journal.values[0],

        'key_passages': []

    }



    for item in record['body_text']:

        sentences = item['text'].split('. ')

        for s in sentences:

            if len(re.findall('|'.join(cv19.PERSISTENCE_SYNONYMS), s.lower())) > 0:

#                 if len(re.findall('|'.join(time_strings), s.lower())) > 0:

                temp_dict['key_passages'].append(s)

    if len(temp_dict['key_passages']) == 0:

        temp_dict['key_passages'] = ['<i>Failed to extract figures - check manually.</i>']

    table_persist.append(temp_dict)

    

table_persist = pd.DataFrame(table_persist)



table_persist['key_passages'] = (table_persist

                                 .key_passages

                                 .apply(lambda x: '<br><br>'.join(x)))

table_persist['title_link'] = table_persist.apply(lambda x: f'<a href="{x.doi}">{x.title}</a> ({x.journal})',

                                                    axis=1)

# table_persist.drop(['title', 'doi'], axis=1, inplace=True)
cv19.display_dataframe(table_persist.loc[table_persist.key_passages

                                         != '<i>Failed to extract figures - check manually.</i>',

                                     ['publication_date',

                                      'authors',

                                      'title_link',

                                      'key_passages']],

                       'Table of Persistence Findings')
meta.to_csv('augmented_metadata_full.csv', index=False)
table_repr.to_csv('reproduction_table.csv', index=False)
table_incub.to_csv('incubation_table.csv', index=False)
table_persist.to_csv('persistence_table.csv', index=False)