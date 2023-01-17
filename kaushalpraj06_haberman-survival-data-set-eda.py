import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
"""Downloaded Haberman Dataset from https://www.kaggle.com/gilsousa/haberman-s-survival"""

#Load Dataset in pandes DataFrame
hb = pd.read_csv("../input/haberman.csv",header=None,names=['Age','TreatmentYear','NoOfNodes','Survival'])
#Data-point and Features in Haberman Dataset
print(hb.shape)
#Columns present 
print(hb.columns)
hb.describe()
#Dataset of each class
print(hb['Survival'].value_counts())
#PairPlot
sns.set_style('whitegrid')
sns.pairplot(hb,hue='Survival',vars=['Age','TreatmentYear','NoOfNodes'])
plt.show()
#Distribution Plot for Age
sns.set_style('whitegrid')
sns.FacetGrid(hb,hue='Survival',size=5).map(sns.distplot,'Age').add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(hb,hue='Survival',size=5).map(sns.distplot,'TreatmentYear').add_legend()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(hb,hue='Survival',size=5).map(sns.distplot,'NoOfNodes').add_legend()
plt.show()
hb_1 = hb[hb['Survival']==1]
hb_2 = hb[hb['Survival']==2]
counts,bin_edges = np.histogram(hb_1['NoOfNodes'],bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='Survived-PDF')
plt.plot(bin_edges[1:],cdf,label='Survived-CDF')

counts,bin_edges = np.histogram(hb_2['NoOfNodes'],bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='Not Survived-PDF')
plt.plot(bin_edges[1:],cdf,label='Not Survived-CDF')
plt.legend()
plt.xlabel('No of Nodes')
plt.show()
counts,bin_edges = np.histogram(hb_1['Age'],bins=10,density=True)
pdf = counts/sum(counts)
cdf =np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='Survived-PDF')
plt.plot(bin_edges[1:],cdf,label='Survived-CDF')

counts,bin_edges = np.histogram(hb_2['Age'],bins=10,density=True)
pdf = counts/sum(counts)
cdf =np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='Not-Survived-PDF')
plt.plot(bin_edges[1:],cdf,label='Not-Survivrd-CDF')
plt.legend()
plt.xlabel('Age')
plt.show()
counts,bin_edges = np.histogram(hb_1["TreatmentYear"],bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(counts)
plt.plot(bin_edges[1:],pdf,label='Survived PDF')
plt.plot(bin_edges[1:],cdf,label='Survived CDF')

counts,bin_edges = np.histogram(hb_2['TreatmentYear'],bins=10,density=True)
pdf = counts/sum(counts)
cdf = np.cumsum(counts)
plt.plot(bin_edges[1:],pdf,label='Not-Survived PDF')
plt.plot(bin_edges[1:],cdf,label='Not-Survived CDF')
plt.legend()
plt.xlabel('Treatment Year')
plt.show()


sns.boxplot(x='Survival',y='NoOfNodes',data=hb)
plt.show()
sns.violinplot(x='Survival',y='NoOfNodes',data=hb,size=8)
plt.show()
