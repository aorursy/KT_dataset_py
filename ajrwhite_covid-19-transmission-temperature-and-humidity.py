import pandas as pd

import covid19_tools as cv19

import os

import re

from IPython.core.display import display, HTML

import html



pd.set_option('display.max_columns', 500)
# Load the metadata

METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'

meta = cv19.load_metadata(METADATA_FILE)

meta, covid19_counts = cv19.add_tag_covid19(meta)



# Load research design

meta2 = pd.read_csv('../input/cord-19-study-metadata-export/metadata_study.csv')

meta2['design'] = meta2.Design.apply(lambda x: cv19.DESIGNS[x])

meta = meta.merge(meta2[['Id', 'design']], left_on='sha', right_on='Id', how='left')

meta['design'] = meta.design.fillna('Other')

meta.drop('Id', axis=1, inplace=True)



print('Loading full text for tag_disease_covid19')

full_text = cv19.load_full_text(meta[meta.tag_disease_covid19],

                                '../input/CORD-19-research-challenge/')



full_text_df = pd.DataFrame(full_text)



case_sensitive_keywords = [

    r'^\b$' # regex for match nothing'

]

case_insensitive_keywords = [

    'air temperature',

    'ambient temperature',

    'humidity',

    'effect of temperature',

    'role of temperature',

    'climate'

]



airtemp_df = full_text_df[full_text_df.body_text.astype(str)

                          .str.lower()

                          .str.contains('|'.join(case_insensitive_keywords)) |

                          full_text_df.body_text.astype(str)

                          .str.contains('|'.join(case_sensitive_keywords))]



analysis_df = cv19.term_matcher(airtemp_df, meta,

                                case_sensitive_keywords,

                                case_insensitive_keywords)



temp_id = []

html_string = f'<h1>Relevant papers</h1><p><b>{len(airtemp_df)} papers found</b><br><br>'

for i, row in enumerate(analysis_df.itertuples()):

    current_id = [row.doi, row.authors, row.title]

    if current_id != temp_id:

        temp_id = current_id

        if i > 0:

            html_string += '</ul><br><br>'

        html_string += f'<b><a href="{row.doi}">{html.escape(row.title)}</a></b><br>'

        html_string += f'{row.authors} ({row.publish_time})<br>'

        html_string += f'<i>Research design: {row.design}</i>'

        html_string += '<ul>'

    html_string += f'<li>{html.escape(row.extracted_string)}</li>'

html_string += '</ul'

display(HTML(html_string))
analysis_df.to_csv('transmission_airtemp.csv', index=False)