import pandas as pd

df=pd.read_csv('../input/training_variants')

dfTest=pd.read_csv('../input/test_variants')
dfSum=pd.concat([df,dfTest])
dfSum['Gene_And_Variation']=dfSum['Gene']+' '+dfSum['Variation']

dfSum.head()
dfSum.info()
dfSum['Gene_And_Variation'].value_counts().head()
dfSum['Gene_And_Variation'].value_counts().max()
import numpy as np



def IsFusion(s):

    n = np.array(s.split()).size

    lastWord=s.split()[n-1]

    if (lastWord.lower()=='fusion'): return True

    return False



df['IsFusion']=df['Variation'].apply(lambda s: IsFusion(s))

df[df['IsFusion']==1].head()
df[df['Variation']=='TMPRSS2-ETV1 Fusion']
df[df['Variation']=='EWSR1-ETV1 Fusion']
df[df['Variation']=='ETV6-NTRK3 Fusion']