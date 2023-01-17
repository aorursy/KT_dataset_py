import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math





df_type_chart = pd.read_csv('../input/type-chart.csv')

# Remove second type for the defender

df_type_chart['defense-type2'].fillna(0, inplace=True)

df_type_chart = df_type_chart[df_type_chart['defense-type2'] == 0]

df_type_chart = df_type_chart.drop(df_type_chart['defense-type2'])

print(df_type_chart.head())



df_moves = pd.read_csv('../input/moves.csv')

print(df_moves.head())
for i in range(10):

    print(df_moves.iloc[i]['move'])