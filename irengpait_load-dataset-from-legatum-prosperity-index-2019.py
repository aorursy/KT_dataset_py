import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import json
with open('../input/Prosperity.json') as f:

    data = json.load(f)
df = pd.DataFrame(data['children'])

df.sample(5)
data['sub_indexes'].keys()
legpros_dataset = df[['country', 'isocode', 'safe', 'pers', 'gove', 'soci', 'inve', 'ente', 'mark', 'econ', 'livi', 'heal', 'educ', 'natu']]
legpros_dataset.sample(5)
legpros_dataset.to_csv('Legatum Prosperity Index Scores 2019.csv')
legpros_dataset.to_excel('Legatum Prosperity Index Scores 2019.xlsx')