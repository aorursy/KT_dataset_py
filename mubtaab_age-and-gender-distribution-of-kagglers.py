import pandas as pd

import numpy as np

import seaborn as sb
df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding = 'ISO-8859-1')

df[:10]
dfg = df.groupby(['GenderSelect', 'Age']).size().reset_index()

dfg['GenderSelect'].value_counts()
import matplotlib.pyplot as plt

total = dfg[0].sum()



p1 = plt.bar(dfg.loc[dfg['GenderSelect'] == 'Male']['Age']

             ,dfg.loc[dfg['GenderSelect'] == 'Male'][0]

             , color='#0F496B')



p1 = plt.bar(dfg.loc[dfg['GenderSelect'] == 'Female']['Age']

             ,dfg.loc[dfg['GenderSelect'] == 'Female'][0]

            , color='#E41B17')



p1 = plt.bar(dfg.loc[dfg['GenderSelect'] == 'A different identity']['Age']

             ,dfg.loc[dfg['GenderSelect'] == 'A different identity'][0]

             , color='#E2CC5F')



p1 = plt.bar(dfg.loc[dfg['GenderSelect'] == 'Non-binary, genderqueer, or gender non-conforming']['Age']

             ,dfg.loc[dfg['GenderSelect'] == 'Non-binary, genderqueer, or gender non-conforming'][0]

             , color='#E2CC5F')
dfgf = dfg[(dfg['GenderSelect'] == 'Male' )|(dfg['GenderSelect'] == 'Female')]

dfgf = dfgf[(dfgf['Age'] > 18) & (dfgf['Age'] < 60)]

dfgfp = dfgf.pivot(index='Age', columns='GenderSelect', values=0)

for i in dfgfp.columns:

    dfgfp[i] = dfgfp[i]/(dfgfp['Female']+dfgfp['Male'])

dfgfp





dfgfp['Male'].index



p1 = plt.bar(dfgfp.index

             ,dfgfp['Male']

            , color='#0F496B'

            , alpha = 0.3)



p1 = plt.bar(dfgfp.index

             ,dfgfp['Female']

            , color='#E41B17')