import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
overall_data = pd.read_csv('../input/guns.csv')

suicide_data = overall_data[overall_data.intent == 'Suicide']

homicide_data = overall_data[overall_data.intent == 'Homicide']
data_sources = [overall_data, suicide_data, homicide_data]

titles = ['Overall', 'Suicide', 'Homicide']



interested_columns = ['year', 'month', 'intent', 'sex', 'age', 'race', 'place', 'education']

for i, d in enumerate(data_sources):

    for col in interested_columns:

        if col == 'age' or col == 'month':

            d[col].value_counts().sort_index().plot(kind = 'line')

        elif col == 'year' or col == 'education':

            d[col].value_counts().sort_index().plot(kind = 'bar')

        else:

            d[col].value_counts().plot(kind = 'bar')

        plt.title(titles[i] + ': ' + col)

        plt.show()