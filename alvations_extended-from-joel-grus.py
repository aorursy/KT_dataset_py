# From https://twitter.com/joelgrus/status/1083563799615922176 

import pandas as pd
import json
import requests

from collections import Counter
from itertools import chain

from nltk import word_tokenize
import numpy as np
#url = 'https://www.sutterhealth.org/for-patients/chargemaster-2019.json'
#raw_data = requests.get(url).content

with open('../input/chargemaster-2019.json', 'rb') as fin:
    df = pd.DataFrame(json.loads(fin.read())['CDM'])
df.columns = [col.lower() for col in df.columns]
df.rename(columns={'descripion': 'description'}, inplace=True)
df.head()
Counter(df['hospital_name'])
set(df[df['hospital_name'].isnull()]['facility'])
Counter(chain(*df['description'].str.lower().apply(word_tokenize))).most_common()
