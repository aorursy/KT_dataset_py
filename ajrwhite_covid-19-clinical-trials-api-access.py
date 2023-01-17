import requests

import json

import pandas as pd

import datetime
CDC_BASE_URL = 'https://clinicaltrials.gov/api/query/study_fields?expr=COVID-19&max_rnk=1000&fmt=json'
cdc_extract_fields = [

    'BriefTitle',

    'DesignAllocation',

    'DesignMasking',

    'DesignMaskingDescription',

    'InterventionName',

    'InterventionType',

    'LastKnownStatus',

    'OfficialTitle',

    'OutcomeAnalysisStatisticalMethod',

    'OutcomeMeasureTimeFrame',

    'SecondaryOutcomeMeasure',

    'StartDate',

    'StudyFirstPostDate',

    'StudyFirstPostDateType',

    'StudyFirstSubmitDate',

    'StudyFirstSubmitQCDate',

    'StudyPopulation',

    'StudyType',

    'WhyStopped'

]
query_url = f'{CDC_BASE_URL}&fields={",".join(cdc_extract_fields)}'

print(query_url)
r = requests.get(query_url)
# Check we have a successful extract with code 200

r.status_code
# Load the JSON data to a dictionary

j = json.loads(r.content)
# This is quite a flat JSON structure, so can be loaded into a DataFrame

df = pd.DataFrame(j['StudyFieldsResponse']['StudyFields'])
# Some of the fields are single-item lists which can be cleaned

def de_list(input_field):

    if isinstance(input_field, list):

        if len(input_field) == 0:

            return None

        elif len(input_field) == 1:

            return input_field[0]

        else:

            return '; '.join(input_field)

    else:

        return input_field
for c in df.columns:

    df[c] = df[c].apply(de_list)
df['StudyFirstPostDate'] = pd.to_datetime(df.StudyFirstPostDate)

df = df.sort_values(by='StudyFirstPostDate', ascending=False)
df[df.StudyType == 'Interventional'].head(100)
timestamp = datetime.datetime.now().date().isoformat()



# Write all results

df.to_csv(f'covid19_clinical_trials_{timestamp}.csv', index=False)



# Write just interventional trials

df[df.StudyType ==

   'Interventional'].to_csv(f'covid19_interventional_clinical_trials_{timestamp}.csv',

                                            index=False)