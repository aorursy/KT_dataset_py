# From https://twitter.com/joelgrus/status/1083563799615922176 

import pandas as pd
import json
import requests


#url = 'https://www.sutterhealth.org/for-patients/chargemaster-2019.json'
#raw_data = requests.get(url).content

with open('../input/chargemaster-2019.json', 'rb') as fin:
    df = pd.DataFrame(json.loads(fin.read())['CDM'])
df.columns = [col.lower() for col in df.columns]
df.rename(columns={'descripion': 'description'}, inplace=True)
df.head()
