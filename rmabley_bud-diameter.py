import os

import pandas as pd

# pd.set_option('display.max_columns', None)  # or 1000

# pd.set_option('display.max_rows', 20)  # or 1000

# pd.set_option('display.max_colwidth', -1)  # or 199
df = pd.read_excel('/kaggle/input/Raw Data.xlsx')

df
numBudz = 31

budz = []

for budNum in range (1, numBudz + 1):

    budz.append(budNum)

melted = pd.melt(df, id_vars=['Date', 'Variety', 'Block', 'Replicate', 'Plant', 'Label'], value_vars=budz)

melted = melted.rename(columns={"variable": "Spiral bud number", "value": "Bud Diameter"})

melted = melted.sort_values(by=['Date',  'Plant', 'Block', 'Replicate', 'Spiral bud number'])

melted

melted.to_excel("budz.xlsx", index=False)