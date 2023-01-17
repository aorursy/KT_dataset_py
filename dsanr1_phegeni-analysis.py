import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/phenotype-genotype-integrator/PheGenI.csv')
df.sample(5)
df.info()
df['P-Value']
pv = df['P-Value'][df['P-Value'].apply(lambda x: isinstance(x, str))]
pv = pv.str.split('-')
pv = pd.to_numeric(pv.apply(lambda x: x[0][:-1])) * 10 ** (-pd.to_numeric(pv.apply(lambda x: x[1]), downcast='float'))
pv
df['P-Value'][pv.index] = pv.values
df['P-Value']
pd.to_numeric(df['P-Value'])
df['P-Value'][df['P-Value'].apply(lambda x: isinstance(x, str))]
df['P-Value'].sort_values()
df['P-Value'].sort_values().reset_index()['P-Value'][:].plot(figsize=(15,7))
df['P-Value'].sort_values().reset_index()['P-Value'][:134000].plot(figsize=(15,7))
df['Trait'].unique().size
df.groupby('Context').count().sort_values('Gene', ascending=False)
plt.figure(figsize=(15,5))

sns.countplot(x='Context', data=df, order=df['Context'].value_counts().index)
df[(df['Gene'] == df['Gene 2'])].groupby('Context').count().sort_values('Gene', ascending=False)
(df['Gene'] == df['Gene 2']).sum()
plt.figure(figsize=(15,5))

sns.countplot(x='Context', data=df[(df['Gene'] == df['Gene 2'])], order=df['Context'].value_counts().index)
df[(df['Gene'] != df['Gene 2'])].groupby('Context').count().sort_values('Gene', ascending=False)
df[(df['P-Value']>0) & (df['P-Value']<10**-300)].sort_values(by='P-Value')
df_p = df[df['P-Value'] < 5 * 10 ** -8]
df_p.groupby('Trait').count().sort_values('P-Value', ascending=False).head(10)
df_p.groupby('Trait').count().sort_values('P-Value', ascending=False)['Gene'].plot(figsize=(18,7))
df_p['Trait'].unique().size
traits = df_p.groupby('Trait').count().sort_values('P-Value', ascending=False).index
traits[:50]
import difflib
matches = difflib.get_close_matches('atherosclerosis', traits, n=15, cutoff=.4)

matches
def genes_by_trait(trait):

    temp = df_p[df_p['Trait']==trait]

    return set(temp['Gene']).union(set(temp['Gene 2']))
len(genes_by_trait('Body Mass Index'))
list_1 = ['Blood Pressure', 'Stroke', 'Diabetes Mellitus','Diabetes Mellitus, Type 2','Diabetes Mellitus, Type 1', 'Myocardial Infarction', 'Atherosclerosis', 'Plaque, Atherosclerotic']
factors_paired = [(i,j) for i in list_1 for j in list_1]
common_genes = []



for i,j in factors_paired:

    common_genes.append(len(genes_by_trait(i).intersection(genes_by_trait(j))))
common_genes = np.array(common_genes).reshape(len(list_1),len(list_1))
common_genes = pd.DataFrame(common_genes, index=list_1, columns=list_1)
common_genes
plt.figure(figsize=(15,5))

common_genes.style.background_gradient(cmap='YlOrRd', axis=0)
common_genes = genes_by_trait('Stroke').intersection(genes_by_trait('Diabetes Mellitus')).intersection(genes_by_trait('Blood Pressure'))

print(len(common_genes))

common_genes
matches = difflib.get_close_matches('inflammatory bowel', traits, n=15, cutoff=.4)

matches
list_2 = ['Multiple Sclerosis', 'Psoriasis', 'Lupus Erythematosus, Systemic', 'Crohn Disease', 'Inflammatory Bowel Diseases', 'Diabetes Mellitus, Type 1']
factors_paired = [(i,j) for i in list_2 for j in list_2]
common_genes = []



for i,j in factors_paired:

    common_genes.append(len(genes_by_trait(i).intersection(genes_by_trait(j))))
common_genes = np.array(common_genes).reshape(len(list_2),len(list_2))
common_genes = pd.DataFrame(common_genes, index=list_2, columns=list_2)
common_genes
plt.figure(figsize=(15,5))

common_genes.style.background_gradient(cmap='YlOrRd', axis=0)
import requests, sys

import pprint


server = "https://rest.ensembl.org"

ext = "/phenotype/gene/homo_sapiens/GCKR?include_associated=0"

     

r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})

     

if not r.ok:

    r.raise_for_status()

    sys.exit()

     

decoded = r.json()

pprint.pprint(decoded)