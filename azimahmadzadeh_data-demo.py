import json

import pandas as pd
p = '/kaggle/input/bigdata2020-flare-prediction-1/train_partition1_data.json'

n = 1

with open(p) as infile:

    for line in infile:  # Each 'line' is one MVTS with its single label ('X', 'M', 'C', 'B', or 'Q').

        n += 1

print('--> There are [{}] mvts in this file.'.format(n-1))
mvts_list = []

n_mvts, i = 5, 0

with open(p) as infile:

    for line in infile:

        i = i + 1

        if n_mvts >= i:

            d: dict = json.loads(line)  # each line is a dictionary

            for k, v in d.items():

                mvts_list.append(v)
mvts = mvts_list[0]

lab = mvts['label']  # class label corresponding to this mvts

values = mvts['values']  # all mvts values

print('--> This mvts is labeled as [{}].'.format(lab))

print('--> Its corresponding mvts looks like this:')

print(values)
param_names = list(values.keys())

print('--> Each mvts keeps track of these [{}] (magnetic-field) parameters:'.format(len(param_names)))

print(param_names)
print('--> Each parameter has [{}] records; a 12-hour window of observation with a 12-minute cadence.'.format(len(values['TOTUSJH'])))

print('--> These are the values for the parameter [{}]:'.format(param_names[0]))

print(values[param_names[0]])
pd.DataFrame.from_dict(values)