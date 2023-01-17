import pandas as pd

import numpy as np

df = pd.read_csv("../input/CAGEandNAICS.csv", na_filter=False)

df.head()
df['naics'] = df.NAICS_CODE_STRING.str.split("~")

df = df.drop('NAICS_CODE_STRING', axis='columns')

df.head()
df[df['naics'].str.contains('^Y', na=False)]
df['naicsY'] = df.apply(lambda x: [], axis=1)

df.head()
for i in range(len(df)):

    df.loc[i, 'naicsLen'] = len(df.loc[i,'naics'])

    for j in range(len(df.loc[i,'naics'])):

        if df.loc[i,'naics'][j].find("Y") > 0 :

            df.loc[i,'naicsY'].append(df.loc[i,'naics'][j])

    df.loc[i, 'naicsLenY'] = len(df.loc[i,'naicsY'])

df
df = df.drop(['naics', 'naicsLen', 'naicsLenY'], axis='columns')

df