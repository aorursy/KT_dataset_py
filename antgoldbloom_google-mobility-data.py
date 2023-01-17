import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('max_rows',200)
df_country = pd.read_csv('/kaggle/input/google-mobility/mobility_google.csv',index_col=0)

df_country = df_country.drop_duplicates()

for cols in df_country.columns:

    df_country[cols] = df_country[cols].map(lambda x: x.lstrip('+').rstrip('%'))

    df_country[cols] = df_country[cols].apply(pd.to_numeric, errors='coerce')



    
df_country[df_country.index.isin(['Italy', 'Spain', 'New Zealand', 'France', 'United Kingdom', 'India', 'Israel', 'United States', 'Australia', 'Japan', 'Sweden', 'South Korea', 'Taiwan','Brazil','Argentina','Norway','Denmark'])].sort_values(by='Retail & recreation',ascending=True)

df_country[df_country.index.isin(['Italy', 'Spain', 'New Zealand', 'France', 'United Kingdom', 'India', 'Israel', 'United States', 'Australia', 'Japan', 'Sweden', 'South Korea', 'Taiwan','Brazil','Argentina','Norway','Denmark'])].sort_values(by='Workplaces',ascending=True)
df_country.sort_values(by='Retail & recreation',ascending=True)