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

meta, riskfac_counts = cv19.add_tag_risk(meta)

n = sum(meta.tag_disease_covid19 & meta.tag_risk_factors)

print(f'{n} papers on tag_disease_covid19 x tag_risk_factors')



# Load research design

meta2 = pd.read_csv('../input/cord-19-study-metadata-export/metadata_study.csv')

meta2['design'] = meta2.Design.apply(lambda x: cv19.DESIGNS[x])

meta = meta.merge(meta2[['Id', 'design']], left_on='sha', right_on='Id', how='left')

meta['design'] = meta.design.fillna('Other')

meta.drop('Id', axis=1, inplace=True)



print('Loading full text for tag_disease_covid19 x tag_risk_factors')

full_text = cv19.load_full_text(meta[meta.tag_disease_covid19 & meta.tag_risk_factors],

                                '../input/CORD-19-research-challenge/')



full_text_df = pd.DataFrame(full_text)



case_sensitive_keywords = [

    r'^\b$' # regex for match nothing

]

case_insensitive_keywords = [

    'alcohol abuse',

    'alcohol addiction',

    'alcohol consumption',

    'alcohol dependence',

    'alcohol withdrawal',

    'alcohol related',

    'alcoholic liver',

    'alcoholism',

    'consumed alcohol',

    'alcohol consumed',

    'chronic alcohol',

    r'\bgout\b',

    'excessive drink',

    'heavy drink',

    'frequent drink',

    'frequency of alcohol',

    'regular drink',

    'regular alcohol',

    'problem drink',

    'drink problem',

    'drinking problem',

    'drinker',

    'binge drink',

    'cirrhosis'

]



alcohol_df = full_text_df[full_text_df.body_text.astype(str)

                          .str.lower()

                          .str.contains('|'.join(case_insensitive_keywords)) |

                          full_text_df.body_text.astype(str)

                          .str.contains('|'.join(case_sensitive_keywords))]



analysis_df = cv19.term_matcher(alcohol_df, meta,

                                case_sensitive_keywords,

                                case_insensitive_keywords)



temp_id = []

html_string = f'<h1>Relevant papers</h1><p><b>{len(alcohol_df)} papers found</b><br><br>'

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
analysis_df.to_csv('risk_factors_drinking.csv', index=False)