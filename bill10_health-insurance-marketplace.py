# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Rate.csv')
df.head().T
df.loc[df.IndividualRate>9000, 'IndividualRate']=None
df.loc[df.IndividualRate==0, 'IndividualRate']=None
df['IndividualRate'].describe()
df.loc[df.IndividualRate==999999.00, 'IndividualRate']=None
df['IndividualRate'].max()
dfi=df[['BusinessYear', 'IssuerId', 'IndividualRate']].groupby(['IssuerId','BusinessYear']).mean().reset_index()
dfi=dfi.pivot(index='IssuerId', columns='BusinessYear', values='IndividualRate')
dfc=df[['BusinessYear', 'IssuerId', 'Couple']].groupby(['IssuerId','BusinessYear']).mean().reset_index()
dfc=dfc.pivot(index='IssuerId', columns='BusinessYear', values='Couple')
dfi=dfi.merge(dfc, how='left', left_index=True, right_index=True, suffixes=('','c'))
dfi.to_csv('rate.tsv',sep='\t')
df2=pd.read_csv('../input/BenefitsCostSharing.csv')
df2.head().T
df3=df2.loc[df2.BenefitName=='Infusion Therapy',['BenefitName','BusinessYear','IssuerId']]
df3=df3.drop_duplicates(['IssuerId','BusinessYear'])
df3=df3.pivot(index='IssuerId', columns='BusinessYear', values='BenefitName')
df3[~df3.isnull()]='Yes'
df3=df3.fillna('No')
df4=df2.loc[df2.BenefitName=='Chemotherapy',['BenefitName','BusinessYear','IssuerId']]

df4=df4.drop_duplicates(['IssuerId','BusinessYear'])

df4=df4.pivot(index='IssuerId', columns='BusinessYear', values='BenefitName')

df4[~df4.isnull()]='Yes'

df4=df4.fillna('No')
df3=df3.merge(df4, how='left', left_index=True, right_index=True, suffixes=('','c'))
df3.to_csv('cancer.tsv',sep='\t')