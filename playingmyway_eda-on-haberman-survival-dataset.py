#Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Load data

data=pd.read_csv("../input/haberman.csv",header=None, names=['Age', 'Op_Year', 'axil_nodes_det', 'Surv_status'])

data.shape
data.columns
# Printing few data points
data.head()
# Calculate no of classes and data points per class
data['Surv_status'].value_counts()

#Data Analysis
data.describe()
data['Op_Year'].value_counts()
data['axil_nodes_det'].value_counts()
data.plot(kind='scatter',x='axil_nodes_det',y='Surv_status')
plt.show()
#Pair plot

plt.close();
sns.set_style("whitegrid")
sns.pairplot(data,hue='Surv_status',vars=["Age", "Op_Year","axil_nodes_det"],height=3);
plt.show()
#Histogram(1-D sctter plot kind of)

sns.FacetGrid(data,hue="Surv_status",size=5)\
    .map(sns.distplot, "axil_nodes_det")\
    .add_legend();
plt.ylabel('Frequency')
plt.show();

#PDF AND CDF For feature Age

data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['Age'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')


data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['Age'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)

plt.plot(bin_edges_2[1:],pdf_2,label='pdf_not-survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_not-survived')


plt.xlabel("Age")
plt.ylabel("Probabilty")
plt.legend()
plt.title("pdf and cdf of Age")
plt.show();

#PDF AND CDF For feature Op_Year
plt.close();
data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['Op_Year'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.figure(1)


plt.subplot(211)
plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')
plt.xlabel("Op_Year")
plt.ylabel("Probabilty")
plt.legend()
plt.title("Pdf and cdf plot of Op_Year")



data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['Op_Year'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)

plt.subplot(212)
plt.plot(bin_edges_2[1:],pdf_2,label='pdf_not-survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_not-survived')



plt.xlabel("Op_Year")
plt.ylabel("Probabilty")
plt.legend()

plt.show();

#PDF AND CDF For feature axil_nodes_det

data_1=data[data['Surv_status']==1]
counts, bin_edges= np.histogram(data_1['axil_nodes_det'],bins=10, density= True)
pdf=counts/(sum(counts))
print(pdf)

print(bin_edges)

cdf=np.cumsum(pdf)
print(cdf)

plt.plot(bin_edges[1:],pdf,label='pdf_survived')
plt.plot(bin_edges[1:],cdf,label='cdf_survived')


data_2=data[data['Surv_status']==2]
counts_2, bin_edges_2= np.histogram(data_2['axil_nodes_det'],bins=10, density= True)
pdf_2=counts_2/(sum(counts_2))
print(pdf_2)

print(bin_edges_2)

cdf_2=np.cumsum(pdf_2)
print(cdf_2)
plt.plot(bin_edges_2[1:],pdf_2,label='pdf_survived')
plt.plot(bin_edges_2[1:],cdf_2,label='cdf_survived')



plt.xlabel("axil_nodes_det")
plt.ylabel("Probabilty")
plt.legend()
plt.title("Pdf and cdf plot of axil_nodes_det")
plt.show();

#BoxPlot for Age
sns.boxplot(x="Surv_status",y="Age",data=data)
plt.show()
#BoxPlot for Op_Year
sns.boxplot(x="Surv_status",y="Op_Year",data=data)
plt.show()
#BoxPlot for axil_nodes_det
sns.boxplot(x="Surv_status",y="axil_nodes_det",data=data)
plt.show()
#violinPlot for axil_nodes_det
sns.violinplot(x="Surv_status",y="axil_nodes_det",data=data)
plt.show()
#violinPlot for Op_Year
sns.violinplot(x="Surv_status",y="Op_Year",data=data)
plt.show()
#violinPlot for AgeS
sns.violinplot(x="Surv_status",y="Age",data=data)
plt.show()