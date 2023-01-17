import covid19_tools as c19

import cdc_clinical_trials_api as cdc



import pandas as pd

import numpy as np

import datetime



from IPython.core.display import display, HTML

import html



api_output = cdc.run_query()

clinical_trials = cdc.json_to_df(api_output)

intervent = clinical_trials[clinical_trials.StudyType == 'Interventional'].copy()



stop_words = ['of', 'and', 'is', 'in', 'for', 'at', 'or',

              'standard', 'care', 'oral', 'treatment',

              'tablet', 'tablets', 'human', 'injection', 'therapy',

              'placebo', 'placebos', 'cells', 'mg', 'with', 'usual',

              'inhalation', 'a', 'group', 'control', 'gas', '1', '2',

              'iv', 'intravenous', 'recombinant', 'health', 'sterile',

              'information', 'solution', 'medicine', 'medicines',

              'sulfate', 'vaccine', 'normal', 'blood', 'dose',

              'convalescent', 'infusion', 'combined', 'training',

              'medical', 'practice', 'phosphate', 'day', 'days',

              'hydrochloride', 'saline', 'device', 'chinese',

              'intensive', 'therapy', 'high', 'best', 'sars-cov',

              'sars', 'cov', 'based', 'derived', 'low', 'anti',

              'alpha', 'beta', 'non', 'use', 'n', 'i', 'granules',

              'via'

             ]



therapies = (intervent.InterventionName

             .str.lower()

             .str.replace('vitamin ', 'vitamin_') # Trick for joining vitamins

             .str.replace(' acid', '_acid')

             .str.replace(' blocker', '_blocker')

             .str.replace(r'uc\-mscs|uc mscs', 'mscs')

             .str.replace('car-nk', 'nk')

             .str.replace(r'\b\d+\b', ' ')

             .str.replace('abidol', 'arbidol') # This appears to be a typo in the data

             .str.replace('stem ', 'stem_')

             .str.replace('nitric oxide', 'nitric_oxide') # Join compound names

             .str.replace(r';|/|,|\.|\+|-|:|%|\(|\)', ' ')

             .str.replace('|'.join([f'\\b{sw}\\b' for sw in stop_words]), ' ')

             .str.split(expand=True)

             .stack()

             .str.strip()

             .value_counts())



for therapy in therapies[therapies > 2].index:

    intervent.loc[intervent.InterventionName

                        .str.lower()

                        .str.replace('-', ' ')

                        .str.contains(r'\b' + therapy.replace('_', ' ')),

                  f'tag_therapy_{therapy}'

                  ] = True

    intervent[f'tag_therapy_{therapy}'] = intervent[f'tag_therapy_{therapy}'].fillna(False)
html_string = ''

abbr = ['nk', 'mscs']

for therapy in therapies[therapies > 2].index:

    relevant_trials = intervent[intervent[f'tag_therapy_{therapy}']]

    if therapy in abbr:

        therapy = therapy.upper()

    else:

        therapy = therapy.title()

    html_string += f'<p><h2>{therapy.replace("_", " ")}</h2></p>'

    html_string += f'{len(relevant_trials)} clinical trials found<br>'

    html_string += '<table><tr><th>Title</th><th>Interventions</th>'

    html_string += '<th>Allocation</th><th>Masking</th><th>First Submission Date</th></tr>'



    for row in relevant_trials.itertuples():

        html_string += f'<tr>'

        html_string += f'<td>{html.escape(row.OfficialTitle)}</td>'

        html_string += f'<td>'

        for inv in row.InterventionName.split('; '):

            html_string += f'{inv}<br>'

        html_string += '</td>'

        try:

            html_string += f'<td>{html.escape(row.DesignAllocation)}</td>'

        except:

            html_string += '<td></td>'

        try:

            html_string += f'<td>{html.escape(row.DesignMasking)}</td>'

        except:

            html_string += '<td></td>'

        try:

            html_string += f'<td>{html.escape(row.StudyFirstSubmitDate)}</td>'

        except:

            html_string += '<td></td>'

        html_string += '</tr>'

    html_string += '</table><br><br>'

display(HTML(html_string))
meta = c19.load_metadata('../input/CORD-19-research-challenge/metadata.csv')

meta, _ = c19.add_tag_covid19(meta)

meta = meta[meta.tag_disease_covid19]

print(f'filtering down to {len(meta)} papers on Covid-19')
full_text = c19.load_full_text(meta, '../input/CORD-19-research-challenge')
full_text_df = pd.DataFrame(full_text)
html_string = ''

for therapy in therapies[therapies > 2].index:

    therapy_df = full_text_df[full_text_df.body_text.astype(str)

                              .str.lower()

                              .str.contains(r'\b' + therapy.replace('_', ' ') + r'\b')]

    html_string += f'<h2>{therapy.title()}</h2>'

    html_string += f'<i>found {len(therapy_df)} papers</i><br>'



    for row in therapy_df.itertuples():

        url = meta[meta.sha == row.paper_id].doi.values[0]

        html_string += f'<b><a href="{url}">{html.escape(row.metadata["title"])}</a></b>, '

        try:

            authors = row.metadata['authors'][0]['last']

            if authors == '':

                authors = 'Authors not listed'

            elif len(row.metadata['authors']) > 1:

                authors += ' et al'

        except:

            authors = 'Authors not listed'

        html_string += f'{authors}'

        html_string += '<ul>'

        for item in row.body_text:

            sentences = item['text'].split('. ')

            for s in sentences:

                if therapy.replace('_', ' ') in s.lower():

                    html_string += f'<li>{html.escape(s)}</li>'

        html_string += '</ul>'

display(HTML(html_string))
timestamp = datetime.datetime.now().date().isoformat()

intervent.to_csv(f'clinical_trials_{timestamp}.csv', index=False)