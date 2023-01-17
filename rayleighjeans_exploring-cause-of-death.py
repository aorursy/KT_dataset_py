import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Read in the death records to pandas df
deaths = pd.read_csv('../input/DeathRecords.csv')
codes = pd.read_csv('../input/Icd10Code.csv')
alc = list(codes[codes['Description'].str.contains('alcohol')]['Code'])
narc = list(codes[codes['Description'].str.contains('narcotics')]['Code'])

print('Alcohol:', deaths[deaths['Icd10Code'].isin(alc)].shape[0])
print('Narcotics:', deaths[deaths['Icd10Code'].isin(narc)].shape[0])
deaths[deaths['MannerOfDeath']==2]['Age'].hist(bins=range(102))
d = pd.merge(deaths,codes,left_on='Icd10Code',right_on='Code')
ed = pd.read_csv('../input/Education2003Revision.csv')
print(ed['Description'])
d[d['Icd10Code'].str.contains('Y35')]['Education2003Revision'].hist(bins=range(8))
count = d[d['Icd10Code']>='R']['Description'].value_counts()
pd.set_option('display.max_rows',1000)
pd.set_option('display.max_colwidth',100)
count[count.values>=12].sort_values(ascending=True).to_frame()