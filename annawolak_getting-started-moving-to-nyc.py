import pandas as pd

# Display all columns instead of shortening with ellipses

pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/median_price.csv')



df.head()
nyc = df.loc[df['City'] == 'New York']

nyc = nyc[['RegionName', '2016-09']]

nyc = nyc.sort(['2016-09'])

nyc = nyc.set_index('RegionName')

nyc
nyc.plot(kind='bar', sort_columns=True)