# This Python 3 environment is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.

import os

import json

import os

from collections import defaultdict as dd

import pandas as pd



counter = dd(list)

counter_files = 0

counter_empty_docs = 0



# run on last version

path = '/kaggle/input/cord19-expertsystem-mesh/cord19_expertsystem_mesh_060320'

for dirname, _, filenames in os.walk(path):

    if 'json' in dirname:

        print(f"{'/'.join(dirname.split('/')[-2:])} has {len(filenames)} files")

    for filename in filenames:

        if not filename.endswith('.json'):

            continue

        with open(os.path.join(dirname, filename), 'r') as si:

            json_data = json.loads(si.read())

            counter_empty_keys = []

            counter_files += 1

            for key in json_data:

                if json_data[key]:

                    # we are only interested in knowing if the information is present

                    if key in ('language', 'cord_uid', 'paper_id'):

                        counter[key].append(1)

                    # for other fields, we want to know how many extractions for the current paper

                    else:

                        counter[key].append(len(json_data[key]))

                else:

                    counter_empty_keys.append(key)

            if len(counter_empty_keys) >= 4:

                counter_empty_docs += 1

                

print(f"Total files: {counter_files}")
from pprint import pprint

pprint(json_data)
pprint(json_data.keys())
data = dd(list)

headers = ['field', 'presence in files', 'extractions (sum)', 'extractions (mean)']

for field, extractions in counter.items():

    sum_total_extractions = sum(extractions)

    total_extractions = len(extractions)

    mean_extractions = f"{sum_total_extractions / total_extractions:.2f}"

    availability = f"{len(extractions) / counter_files * 100:.2f}"



    contents = [field, availability, sum_total_extractions, mean_extractions]

    for header, value in zip(headers, contents):

        data[header].append(value)



df = pd.DataFrame(dict(data))

df['extractions (sum)'] = df.apply(lambda x: "{:,.0f}".format(x['extractions (sum)']), axis=1)

print(df.head(10))

        