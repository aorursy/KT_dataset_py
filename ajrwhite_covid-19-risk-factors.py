import covid19_tools as cv19

import pandas as pd

import re

from IPython.core.display import display, HTML

import html



METADATA_FILE = '../input/CORD-19-research-challenge/metadata.csv'



# Load metadata

meta = cv19.load_metadata(METADATA_FILE)

# Add tags

meta, covid19_counts = cv19.add_tag_covid19(meta)

meta, riskfac_counts = cv19.add_tag_risk(meta)



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
print('Loading full text for tag_disease_covid19 x risk factor terminology')

full_text_risk = cv19.load_full_text(meta[meta.tag_disease_covid19 &

                                          meta.tag_risk_factors],

                                     '../input/CORD-19-research-challenge')
# Manual list for highlighting

# Need to look at automated / updateable approach to identifying these

risk_factors = [

    'diabetes',

    'hypertension',

    'smoking',

    'cardiovascular disease',

    'chronic obstructive pulmonary disease',

    'cerebrovascular disease',

    'kidney disease',

    ' age ',

    ' aged',

    'blood type',

    'hepatitis',

    ' male ',

    ' female ',

    ' males ',

    ' females ',

    'arrhythmia',

    ' sex ',

    ' gender ',

    'acute respiratory distress syndrome',

    'sepsis shock',

    'cardiac injury',

    'acute kidney injury',

    'liver dysfunction',

    'gastrointestinal haemorrhage',

    'conjunctivitis',

    'comorbidity',

    'comorbidities',

    'co-morbidity',

    'co-morbidities',

    ' smoker',

    'non-smoker'

]
case_sen_risk_re = [

    r'\bOR\b',

    r'\bCI\b',

    r'\bHR\b' # This is picking up Heart Rate in some papers

]

case_ins_risk_re = [

    'odds ratio',

    'risk ratio',

    'confidence interval',

    'hazard ratio',

    'relative risk',

    'p =',

    'p=',

    'p<',

    'p <',

    'adjusting for',

    'adjusted for',

    'controlling for',

    'controlled for',

    'incidence of'

]

case_sen_risk_re = '|'.join(case_sen_risk_re)

case_ins_risk_re = '(?i)' + '|'.join(case_ins_risk_re)
def highlight_numbers(s):

    match_sequence = re.findall(r'\d+\.\d+', s)

    for ms in list(set(match_sequence)):

        s = re.sub(ms, f'<span style="background-color:#ddf;">{ms}</span>', s)

    return s
display_output = []

csv_data = []

for i, item in enumerate(full_text_risk):

    temp_output = {}

    

    doi = meta[meta.sha == item['paper_id']].doi.values[0]

    temp_output['doi'] = doi

    doi = html.escape(doi)

    

    try:

        authors = item['metadata']['authors'][0]['last']

        if len(item['metadata']['authors']) > 1:

            authors += ' et al'

    except:

        authors = 'No author listed'

    temp_output['authors'] = authors

    authors = html.escape(authors)

    

    title = item['metadata']['title']

    temp_output['title'] = title

    title = html.escape(title)

    

    publish_time = meta[meta.sha == item['paper_id']].publish_time.values[0]

    temp_output['publish_time'] = publish_time

    publish_time = html.escape(publish_time)



    display(HTML(f'Result {i+1}<br><b><i><a href="{doi}">{title}</a></i>, {authors}</b><br>{publish_time}'))

    output_list = []

    csv_output_list = []

    for bt in item['body_text']:

        sentence_list = bt['text'].split(r'. ')

        for s in sentence_list:

            if (len(re.findall(case_sen_risk_re, s)) > 0 or

                len(re.findall(case_ins_risk_re, s)) > 0):

                csv_string = s

                html_string = html.escape(s)

                for rf in risk_factors:

                    html_string = re.sub(rf, f'<mark>{rf}</mark>', html_string)

                html_string = highlight_numbers(html_string)

                output_list.append(html_string)

                csv_output_list.append(csv_string)

    if len(output_list) > 0:

        display(HTML('<ul><li>' + '</li><li>'.join(output_list) + '</li></ul><br><br>'))

        temp_output['key_passages'] = '\n'.join(csv_output_list)

    csv_data.append(temp_output)
risk_factors_df = pd.DataFrame(csv_data)

risk_factors_df.head()
risk_factors_df.to_csv('risk_factors.csv', index=False)