import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
haberman=pd.read_csv("../input/haberman.csv")
haberman.head()
#Adding column name to dataset
columns=['Age','OperationYear','AuxillaryNode','Survival']
haberman_data=pd.read_csv('../input/haberman.csv',names=columns)
haberman_data.head()
haberman_data.shape
haberman_data.Survival.value_counts()

sns.FacetGrid(haberman_data, hue="Survival", size=7) \
   .map(sns.distplot, "Age") \
   .add_legend()
plt.show()

sns.FacetGrid(haberman_data, hue="Survival", size=7) \
   .map(sns.distplot, "OperationYear") \
   .add_legend()
plt.show()

sns.FacetGrid(haberman_data, hue="Survival", size=7) \
   .map(sns.distplot, "AuxillaryNode") \
   .add_legend()
plt.show()
less_than_five = haberman_data.loc[haberman_data["Survival"]==2]
more_than_five = haberman_data.loc[haberman_data["Survival"]==1]


plt.subplot(1, 2, 1)
counts, bin_edges = np.histogram(less_than_five['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf  Survived',
            'Cdf  Survived'])

plt.subplot(1, 2, 2)
counts, bin_edges = np.histogram(more_than_five['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf Died',
            'Cdf Died'])

plt.show()




plt.subplot(1, 2, 1)
counts, bin_edges = np.histogram(less_than_five['OperationYear'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf  Survived',
            'Cdf  Survived'])

plt.subplot(1, 2, 2)
counts, bin_edges = np.histogram(more_than_five['OperationYear'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf Died',
            'Cdf Died'])

plt.show()




plt.subplot(1, 2, 1)
counts, bin_edges = np.histogram(less_than_five['AuxillaryNode'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf  Survived',
            'Cdf  Survived'])

plt.subplot(1, 2, 2)
counts, bin_edges = np.histogram(more_than_five['AuxillaryNode'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf Died',
            'Cdf Died'])

plt.show()




(sns.boxplot(x='Survival',y='AuxillaryNode', data=haberman_data))
plt.show()
plt.close()
sns.boxplot(x='Survival',y='Age', data=haberman_data)
plt.show()
sns.boxplot(x='Survival',y='OperationYear', data=haberman_data)
plt.show()


sns.violinplot(x='Survival',y='AuxillaryNode', data=haberman_data)
plt.show()
sns.violinplot(x='Survival',y='Age', data=haberman_data)
plt.show()
sns.violinplot(x='Survival',y='OperationYear', data=haberman_data)
plt.show()
sns.pairplot(haberman_data,hue="Survival",vars = ["Age","AuxillaryNode","OperationYear"])
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(plt.scatter,"Age","AuxillaryNode").add_legend()
plt.show()
haberman_data[haberman_data['Survival']==1].describe()

haberman_data[haberman_data['Survival']==2].describe()
sns.jointplot(x="Age", y="OperationYear", data=haberman_data, kind="kde");
plt.show();