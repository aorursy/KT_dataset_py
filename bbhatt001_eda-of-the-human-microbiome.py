%matplotlib inline  
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

micro=pd.read_csv('../input/project_catalog.csv')
pd.set_option('display.max_rows', None, 'display.max_columns', None)

micro.shape
micro.columns 
micro.head(2)
micro.info()
micro['Gene Count'].describe()
micro_gene_count=micro[micro['Gene Count']==0]
micro_gene_count['NCBI Superkingdom'].value_counts()
micro_no_gene_progress= micro[(micro['Gene Count']==0) & (micro['Project Status']=='In Progress')]
micro_no_gene_progress['NCBI Superkingdom'].value_counts()
micro_no_gene_complete=micro[(micro['Gene Count']==0) & (micro['Project Status']=='Complete')]
micro_no_gene_complete['NCBI Superkingdom'].value_counts()
micro[micro['Gene Count']==8490]
micro['HMP Isolation Body Site'].nunique()
micro['HMP Isolation Body Site'].value_counts()
micro['HMP Isolation Body Site'].value_counts().plot(kind='bar')
plt.title('Distribution of microorganisms in various body sites')
plt.ylabel('Number of different microbes')
plt.xlabel('Human body sites')
plt.title('Diversity of microorganisms at different body sites')
micro['Genus']= micro['Organism Name'].str.split(' ').str[0]
micro['species']=micro['Organism Name'].str.split(' ').str[1]
micro[['Genus','species']].head()
micro['Genus'].nunique()
micro['Genus'].value_counts().head(10)
micro.groupby('NCBI Superkingdom').count()
micro[micro['NCBI Superkingdom']=='Error!!!']
micro['NCBI Superkingdom'].replace('Error!!!', 'Bacteria', inplace=True)
micro[['Domain','NCBI Superkingdom']].isnull().sum()
len(micro.loc[micro['Domain'].isnull()& micro['NCBI Superkingdom'].isnull()])
micro=micro.drop(micro[(micro['Domain'].isnull()) & (micro['NCBI Superkingdom'].isnull())].index)
micro.shape
micro['NCBI Superkingdom'].fillna('NaN', inplace=True)
print(micro.shape)
micro['Domain'] =micro.groupby('NCBI Superkingdom')['Domain'].transform(lambda x: x.fillna(x.mode().max()))
micro['Domain'].isnull().sum()
micro['NCBI Superkingdom']= micro.groupby('Domain')['NCBI Superkingdom'].transform(lambda x: x.replace('NaN', x.mode().max()))
micro.loc[micro['NCBI Superkingdom']=='NaN']
micro.groupby('NCBI Superkingdom')['HMP Isolation Body Site'].nunique().sort_values(ascending=False)
bac=micro.loc[micro['Domain']=='BACTERIAL']
bac['HMP Isolation Body Site'].unique()
bac['HMP Isolation Body Site'].value_counts(ascending=False).plot(kind='bar')
plt.ylabel('Number of different bacteria')
plt.xlabel('Human body sites')
plt.title('Diversity of bacteria at different body sites')
euk=micro.loc[micro['Domain']=='EUKARYAL']
euk['HMP Isolation Body Site'].unique()
euk['HMP Isolation Body Site'].value_counts(ascending=False).plot(kind='bar')
plt.ylabel('Number of different eukaryotes')
plt.xlabel('Human body sites')
plt.title('Diversity of eukaryotes at different body sites')
vir=micro.loc[micro['Domain']=='VIRUS']
vir['HMP Isolation Body Site'].unique()
arc=micro.loc[micro['Domain']=='ARCHAEAL']
arc['HMP Isolation Body Site'].unique()
z=micro.groupby('Genus')['HMP Isolation Body Site'].nunique().sort_values(ascending=False)
y=pd.DataFrame(z)
w=y[y['HMP Isolation Body Site']>4]
print(w)
w.plot(kind='bar')
plt.ylabel('Number of different body sites')
plt.title('Number of habitats for different microorganisms')
staph=micro.loc[micro['Genus']=='Staphylococcus']
staph['HMP Isolation Body Site'].unique()
micro['NCBI Superkingdom'].value_counts()
viruses= micro[micro['NCBI Superkingdom'] =='Viruses']
viruses['Organism Name']
eukaryotes= micro[micro['NCBI Superkingdom']=='Eukaryota']
eukaryotes['Organism Name']
archaea= micro[micro['NCBI Superkingdom']=='Archaea']
archaea['Organism Name']