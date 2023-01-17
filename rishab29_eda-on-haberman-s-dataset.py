import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
names = ['Age', 'Year operation', 'Axillary nodes detected', 'Survival status']
haberman=pd.read_csv('../input/haberman.csv',names=names)
print (haberman.shape)
haberman.columns
haberman.head()
haberman.describe()
haberman.tail()
haberman['Survival status'].value_counts()
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival status", size=5).map(plt.scatter, "Age", "Axillary nodes detected").add_legend();
plt.show();
# max age
print(haberman.Age.max())
#max nodes detected
print(haberman['Axillary nodes detected'].max())
# Now lets plot pair plots.Since the features are 3 so there will be 3c2 total plots ie 3
plt.close()
sns.set_style("whitegrid")
sns.pairplot(haberman,size=5,hue="Survival status",vars=["Age","Year operation" ,"Axillary nodes detected"])
plt.show()
#The first parameter taken here is Age.
survival_status1 = haberman.loc[haberman["Survival status"] == 1];
survival_status2 = haberman.loc[haberman["Survival status"] == 2];
sns.FacetGrid(haberman, hue="Survival status", size=5).map(sns.distplot, "Age").add_legend();
plt.show();
#The third feature is Axillary nodes detected
sns.FacetGrid(haberman,hue="Survival status",size=5).map(sns.distplot, "Axillary nodes detected").add_legend()
plt.show()

haberman["Survival status"][haberman["Axillary nodes detected"]>=30].value_counts()
haberman["Survival status"][haberman["Axillary nodes detected"]>=40].value_counts()
# So Above pdfs only give us age as single feature on the basis of which we can predict few outcomes.

sns.boxplot(x='Survival status',y='Age', data=haberman)
plt.show()

sns.jointplot(x="Age", y="Axillary nodes detected", data=survival_status1, kind="kde");
plt.show();
sns.jointplot(x="Age", y="Axillary nodes detected", data=survival_status2, kind="kde");

counts,bin_edges=np.histogram(survival_status1["Age"],bins=10,density="true")
pdf=counts/sum(counts)

plt.figure(1)
plt.subplot(1,3,1)
cdf=np.cumsum(pdf)
print(counts)
print(bin_edges)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:],cdf,label='cdf')
plt.xlabel("Age")
plt.legend()
plt.subplots_adjust(wspace=0.1)

nodes_counts,nodes_bin_edges=np.histogram(survival_status1["Axillary nodes detected"],bins=10,density="true")
pdf_nodes=nodes_counts/sum(nodes_counts)
plt.figure(1)
plt.subplot(1,3,2)
cdf_nodes=np.cumsum(pdf_nodes)
print(nodes_counts)
print(nodes_bin_edges)
plt.plot(nodes_bin_edges[1:],pdf_nodes,label='pdf')
plt.plot(nodes_bin_edges[1:],cdf_nodes,label='cdf')
plt.xlabel("Axillary nodes detected")
plt.legend()
plt.subplots_adjust(wspace=0.1)

yop_counts,yop_bin_edges=np.histogram(survival_status1["Year operation"],bins=10,density="true")
pdf_yop=yop_counts/sum(yop_counts)
plt.figure(1)
plt.subplot(1,3,3)
cdf_yop=np.cumsum(pdf_yop)
print(yop_counts)
print(yop_bin_edges)
plt.plot(yop_bin_edges[1:],pdf_yop,label='pdf')
plt.plot(yop_bin_edges[1:],cdf_yop,label='cdf')
plt.xlabel("Year of Operation")
plt.legend()
plt.subplots_adjust(wspace=0.1)

plt.show()

#print(survival_status1.shape)
#print(survival_status1["Axillary nodes detected"][survival_status1["Axillary nodes detected"]<20].count)
counts,bin_edges=np.histogram(survival_status2["Age"],bins=10,density="true")
pdf=counts/sum(counts)

plt.figure(1)
plt.subplot(1,3,1)
cdf=np.cumsum(pdf)
print(counts)
print(bin_edges)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:],cdf,label='cdf')
plt.xlabel("Age")
plt.legend()
plt.subplots_adjust(wspace=0.1)

nodes_counts,nodes_bin_edges=np.histogram(survival_status2["Axillary nodes detected"],bins=10,density="true")
pdf_nodes=nodes_counts/sum(nodes_counts)
plt.figure(1)
plt.subplot(1,3,2)
cdf_nodes=np.cumsum(pdf_nodes)
print(nodes_counts)
print(nodes_bin_edges)
plt.plot(nodes_bin_edges[1:],pdf_nodes,label='pdf')
plt.plot(nodes_bin_edges[1:],cdf_nodes,label='cdf')
plt.xlabel("Axillary nodes detected")
plt.legend()
plt.subplots_adjust(wspace=0.1)

yop_counts,yop_bin_edges=np.histogram(survival_status2["Year operation"],bins=10,density="true")
pdf_yop=yop_counts/sum(yop_counts)
plt.figure(1)
plt.subplot(1,3,3)
cdf_yop=np.cumsum(pdf_yop)
print(yop_counts)
print(yop_bin_edges)
plt.plot(yop_bin_edges[1:],pdf_yop,label='pdf')
plt.plot(yop_bin_edges[1:],cdf_yop,label='cdf')
plt.xlabel("Year of Operation")
plt.legend()
plt.subplots_adjust(wspace=0.1)


plt.show()
