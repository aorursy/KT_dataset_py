import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Load in the .csv files as three separate dataframes

global_df = pd.read_csv('../input/global.csv') 

national_df = pd.read_csv('../input/national.csv')

regional_df = pd.read_csv('../input/regional.csv')
print('\n'.join(global_df.columns.values))
global_biggest_religion = []

for col in global_df.columns.values:

    if '_all' in col:

        global_biggest_religion.append(col)

global_biggest_religion.pop(-1)  # pop 'religion_all' 

print('\n'.join(global_biggest_religion))
main_religions = global_df[global_biggest_religion].copy()
main_religions = main_religions.set_index([global_df.iloc[:]['year'][0:].values.tolist()])
main_religions.plot(figsize=(13,8))
percent_international = []

for col in global_religion:

    new_col = col.replace('all', 'percent')

    percent_international.append(new_col)
main_religions_perc = international[percent_international].copy()

main_religions_perc = main_religions_perc.set_index([international.iloc[:]['year'][0:].values.tolist()])
ax = main_religions_perc[percent_international[:4] + 

             [percent_international[5]] + 

             [percent_international[-2]]].plot.bar(colormap='Greens', figsize=(13,8))

ax.set_xlabel("Year")

ax.set_ylabel("Population percent")

ax.set_facecolor('gray')