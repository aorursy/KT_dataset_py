#Importing all the Libraries required for Excercise
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Reading the CSV downloaded from Kaggle
Dataset=pd.read_csv('../input/haberman.csv')

#Giving Column Names to Dataset
Dataset.columns=['age','surgery_year','axil_nodes','status']
print("Shape of Dataset is: ",Dataset.shape)
print("Head of the Dataset is: \n",Dataset.head())
print("Tail of the Dataset is: \n",Dataset.tail())
print("Columns of the Dataset is: \n",Dataset.columns)
print("Applying Max and Min Functinos")
print("**Minimum Age: ",Dataset['age'].min(),'\t**Maximum Age: ',Dataset['age'].max())
print("**Minimum Axils: ",Dataset['axil_nodes'].min(),'\t**Maximum Axils: ',Dataset['axil_nodes'].max())
#Use of Group by:
g=Dataset.groupby('axil_nodes')
for axil_nodes,node_df in g:
    print("Group by: ",axil_nodes)
    print(node_df,'\n')
print("Row with maximum Axil Nodes:\n ",Dataset[Dataset.axil_nodes==Dataset.axil_nodes.max()])
#Descibing the Haberman Dataset
Dataset.describe()
#Mean of all Input Features
print("Mean of feature Age is: ",Dataset['age'].mean())
print("Mean of feature Year is: ",Dataset['surgery_year'].mean())
print("Mean of feature Axils is: ",Dataset['axil_nodes'].mean())
#Median of all Input Features
print("Median of feature Age is: ",Dataset['age'].median())
print("Median of feature Year is: ",Dataset['surgery_year'].median())
print("Median of feature Axils is: ",Dataset['axil_nodes'].median())
#Use of Quantile over Dataset
print(np.percentile(Dataset["age"],np.arange(0, 100, 25)))
print(np.percentile(Dataset["axil_nodes"],np.arange(0, 100, 25)))
print(np.percentile(Dataset["axil_nodes"],np.arange(0, 125, 25)))
for colomn in Dataset.columns[0:3]:
    sns.FacetGrid(Dataset,hue='status',size=5)\
              .map(sns.distplot,colomn)\
              .add_legend()
    plt.grid()
plt.show()
survived=Dataset[Dataset['status']==1]
plt.figure(figsize=(17,5))
sns.set_style('whitegrid')      
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    plt.subplot(1,3,i+1)
    counts,bin_edges= np.histogram(survived[colomn],bins=20,density=True)
    print("--------****",colomn,"****--------")
    pdf=counts/sum(counts)
    print("PDF is: ",pdf)
    print("Edges are: ",bin_edges)

    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(colomn)
not_survived=Dataset[Dataset['status']==2]
plt.figure(figsize=(17,5))
sns.set_style('whitegrid')      
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    plt.subplot(1,3,i+1)
    counts,bin_edges= np.histogram(not_survived[colomn],bins=20,density=True)
    print("--------****",colomn,"****--------")
    pdf=counts/sum(counts)
    print("PDF is: ",pdf)
    print("Edges are: ",bin_edges)

    cdf=np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(colomn)
fig,axes=plt.subplots(1,3,figsize=(15,5))
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    sns.boxplot(x='status',y=colomn,data=Dataset,ax=axes[i])
fig,axes=plt.subplots(1,3,figsize=(15,5))
for i,colomn in enumerate(list(Dataset.columns[:-1])):
    sns.violinplot(x='status',y=colomn,data=Dataset,ax=axes[i])
sns.set_style('whitegrid')
sns.pairplot(Dataset,hue='status',vars=[Dataset.columns[0],Dataset.columns[1],Dataset.columns[2]],size=4)
plt.show()