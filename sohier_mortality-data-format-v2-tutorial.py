import json

import pandas as pd
data_2015 = pd.read_csv('../input/2015_data.csv', nrows=10**4, dtype=object)

with open('../input//2015_codes.json', 'r') as f_open:

    code_maps_2015 = json.load(f_open)
print(data_2015['358_cause_recode'].mode()[0])
data_2015['decoded_358_cause'] = data_2015['358_cause_recode'].apply(

    lambda x: code_maps_2015['358_cause_recode'][x])
print(data_2015['decoded_358_cause'].mode()[0])
code_maps_2015['358_cause_recode']
data_2015.info()