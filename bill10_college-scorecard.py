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
df=pd.read_csv('../input/Scorecard.csv')
df1=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','COSTT4_A']].dropna()

df1=df1.pivot(index='UNITID', columns='Year', values='COSTT4_A')

#df1.to_csv('cost.tsv',sep='\t')
df2=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','md_earn_wne_p6']].dropna()

df2.loc[df2['md_earn_wne_p6']=='PrivacySuppressed','md_earn_wne_p6']=None

df2['md_earn_wne_p6']=pd.to_numeric(df2['md_earn_wne_p6'])

df2=df2.pivot(index='UNITID', columns='Year', values='md_earn_wne_p6')

df2=df2.apply(lambda x: pd.qcut(x, 2, labels=['No','Yes']), axis=0)

df2.to_csv('earnings.tsv', sep='\t')
df2=df.loc[df['CONTROL']=='Public',['UNITID','INSTNM','Year','CDR2']].dropna()

df2=df2.pivot(index='UNITID', columns='Year', values='CDR2')

df2=(df2==0)

df2.to_csv('default.tsv', sep='\t')