import covid19_tools as cv19

import pandas as pd

import numpy as np

import re

import html

from IPython.display import HTML



pd.set_option('display.max_columns', 500)



METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'

# Load metadata

meta = cv19.load_metadata(METADATA_FILE)

# Add Covid-19 disease tag so we can filter out non-Covid19 papers

meta, _ = cv19.add_tag_covid19(meta)

# Load full text

full_text = cv19.load_full_text(meta[meta.tag_disease_covid19],

                                '../input/CORD-19-research-challenge/')

# Convert full text to a DataFrame

df = pd.DataFrame(full_text)
odds_re = re.compile(r'\bOR\b')

odds_number_re = re.compile(r'\bOR\b.{0,10}?(\d+\.\d+)')
# Run on an example string

or_example = 'Example string containing OR = 3.45, and some other text.'

odds_number_re.findall(or_example)
ci_numbers_re = re.compile(r'\bCI\s?\:?\,?=?\s?(\d+\.\d+).{0,5}?(\d+\.\d+)')
# Test on an example

ci_numbers_re.findall('OR = 4.6 and CI = 3.45 to 4.3')
meta.head()
# list for storing extracted values and key passages

df_list = [] # will be converted to a Pandas DataFrame at the end

output_html = ''

prev_paper = ''

paper_count = 1

for row in df.itertuples():

    paper_id = row.paper_id

    bt = row.body_text

    for item in bt:

        for sentence in item['text'].split('. '):

            temp_dict = {'paper_id': paper_id}

            # Check for odds ratio text

            if len(odds_re.findall(sentence)) > 0:

                # Check for a float in the sentence

                or_numbers = odds_number_re.findall(sentence)

                if len(or_numbers) > 0:

                    if paper_id != prev_paper:

                        prev_paper = paper_id

                        if paper_id.startswith('PMC'):

                            temp_meta = meta[meta.pmcid == paper_id]

                        else:

                            temp_meta = meta[meta.sha == paper_id]

                        title = temp_meta.title.values[0]

                        doi = temp_meta.doi.values[0]

                        authors = temp_meta.authors.values[0]

                        output_html += f'<b>{paper_count}. {html.escape(title)}</b><br>'

                        output_html += f'<a href="{doi}">{doi}</a><br>'

                        output_html += f'Authors: {authors}<br>'

                        paper_count += 1

                    or_numbers = [float(orn) for orn in or_numbers]

                    ci_numbers = ci_numbers_re.findall(sentence)

                    ci_numbers = [(float(cin[0]), float(cin[1])) for cin in ci_numbers]

                    output_html += html.escape(sentence) + '<br><br><i>Extracted odds ratios:</i><ul>'

                    for orn in or_numbers:

                        output_html += f'<li>{orn}</li>'

                    output_html += '</ul>'

                    if len(ci_numbers) > 0:

                        output_html += '<i>Extracted confidence intervals:</i><ul>'

                        for cin in ci_numbers:

                            output_html += f'<li>{cin[0]} - {cin[1]}</li>'

                        output_html += '</ul>'

                    if len(ci_numbers) < len(or_numbers):

                        for i in range(len(or_numbers) - len(ci_numbers)):

                            ci_numbers += [(None, None)]

                    if len(or_numbers) < len(ci_numbers):

                        for i in range(len(ci_numbers) - len(or_numbers)):

                            or_numbers += [None]

                    temp_dict['or_numbers'] = or_numbers

                    temp_dict['ci_numbers'] = ci_numbers

                    temp_dict['sentence'] = sentence

                    df_list.append(pd.DataFrame(temp_dict))

display(HTML(output_html))
extracts_df = pd.concat(df_list).reset_index(drop=True)
extracts_df.head()
extracts_df['ci_lower'] = extracts_df.ci_numbers.apply(lambda x: x[0])

extracts_df['ci_upper'] = extracts_df.ci_numbers.apply(lambda x: x[1])

extracts_df = extracts_df.drop('ci_numbers', axis=1)
extracts_df = extracts_df[['paper_id', 'sentence',

                           'or_numbers',

                           'ci_lower', 'ci_upper']].rename(columns={'or_numbers': 'odds_ratio'})
extracts_df.head()
# Need to split these because they key into different fields in meta

pmc_extracts_df = extracts_df[extracts_df.paper_id.str.startswith('PMC')].copy()

pdf_extracts_df = extracts_df[~extracts_df.paper_id.str.startswith('PMC')].copy()
# paper_id -> pmcid

pmc_extracts_df = pmc_extracts_df.merge(meta, left_on='paper_id',

                                       right_on='pmcid', how='left')
# paper_id -> sha

pdf_extracts_df = pdf_extracts_df.merge(meta, left_on='paper_id',

                                        right_on='sha', how='left')
# Concatenate

extracts_df = pd.concat((pmc_extracts_df, pdf_extracts_df)).reset_index(drop=True)
extracts_df.shape
extracts_df.to_csv('extracted_odds_ratios.csv', index=False)