import chardet

import pandas as pd

from fuzzywuzzy import fuzz



with open('../input/2-3-ekim-2019-haberleri-18-gazete/18GazeteHaberleri_2-3Ekim2019.csv', 'rb') as f:

    result = chardet.detect(f.read()) 



t = pd.read_csv('../input/2-3-ekim-2019-haberleri-18-gazete/18GazeteHaberleri_2-3Ekim2019.csv', encoding=result['encoding'])

t['ANAHABER'] = t['SIRA']

for idx, row in t.iterrows():

    for i,r in t[t['SIRA'] > row['SIRA']].iterrows():

        if (t.loc[i].SIRA==t.loc[i].ANAHABER):

            if (fuzz.ratio(t.loc[i].HABERMETNI,t.loc[idx].HABERMETNI))>70:

                t.at[i,'ANAHABER']=t.loc[idx].SIRA

a = t[['SIRA','ANAHABER']]

a.to_csv('fuzzy70.csv', index=False)                