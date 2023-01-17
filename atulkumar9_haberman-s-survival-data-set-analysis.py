import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels import robust
hsd = pd.read_csv('../input/haberman.csv', names = ['Age', 'Year', 'Aux_nodes', 'Sr_stat'])
hsd.head()
# Converting 1 to Survived and 2 to Not Survived strings to make the data more meaningful

hsd['Sr_stat'] = hsd['Sr_stat'].map({1:"Survived", 2:"Not Survived"})
hsd['Sr_stat'] = hsd['Sr_stat'].astype('category')
hsd.head()
hsd.tail()
hsd.shape
hsd.columns
hsd['Sr_stat'].value_counts()
sns.set(style = "whitegrid")
sns.FacetGrid(hsd, hue = "Sr_stat", size = 5) \
   .map(sns.distplot, "Age") \
   .add_legend()
plt.xlabel("age of patient at the time of operation")
plt.show()
sns.FacetGrid(hsd, hue = "Sr_stat", size = 5) \
   .map(sns.distplot, "Year") \
   .add_legend()
plt.xlabel("patient's year of operation")
plt.show()
sns.FacetGrid(hsd, hue = "Sr_stat", size = 5, ylim = (0, 0.55)) \
   .map(sns.distplot, "Aux_nodes") \
   .add_legend()
plt.xlabel("number of positive auxillary nodes")
plt.show()
survived = hsd.loc[hsd["Sr_stat"] == "Survived"]
not_survived = hsd.loc[hsd["Sr_stat"] == "Not Survived"]
#Survived

counts, bin_edges = np.histogram(survived['Aux_nodes'], bins = 10, density = True)
pdf = counts/sum(counts)
print('FOR SURVIVED')
print()
print("PDF: " ,pdf)
print("BIN EDGES: ", bin_edges)
CDF = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], CDF)

#Not Survived
print()
print()
counts, bin_edges = np.histogram(not_survived['Aux_nodes'], bins = 10, density = True)
pdf = counts/sum(counts)
print('FOR NOT SURVIVED')
print()
print("PDF: " ,pdf)
print("BIN EDGES: ", bin_edges)
CDF = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], CDF)
plt.xlabel('number of positive auxillar nodes')
plt.ylabel('PDF/CDF')
plt.legend(['Survived pdf', 'Survived CDF','Not survived pdf', 'Not survived CDF'])
plt.show()
print("FOR SURVIVED")
survived.iloc[:,0:3].describe()
print("FOR NOT SURVIVED")
not_survived.iloc[:,0:3].describe()
print("Median of positive auxillary value is: {0} for SURVIVED".format(np.median(survived["Aux_nodes"])))
print("Median of positive auxillary value is: {0} for NOT SURVIVED".format(np.median(not_survived["Aux_nodes"])))
print("FOR SURVIVED")
print(robust.mad(survived['Aux_nodes']))
print("FOR NOT SURVIVED")
print(robust.mad(not_survived['Aux_nodes']))
sns.boxplot(x = 'Sr_stat', y = 'Aux_nodes', data = hsd)
plt.xlabel("survival Status")
plt.ylabel("number of positive auxillary nodes")
plt.show()
sns.violinplot(x = 'Sr_stat', y = 'Aux_nodes', data = hsd, size = 5)
plt.xlabel("survival status")
plt.ylabel("number of positive auxilary nodes")
plt.show()
sns.violinplot(x = 'Sr_stat', y = 'Year', data = hsd, size = 5)
plt.xlabel("survival status")
plt.ylabel("year of operation")
plt.show()
sns.violinplot(x = 'Sr_stat', y = 'Age', data = hsd, size = 5)
plt.xlabel("survival status")
plt.ylabel("Age at time of operation")
plt.show()
sns.FacetGrid(hsd, hue = 'Sr_stat', size = 5) \
   .map(plt.scatter, 'Year', 'Aux_nodes') \
   .add_legend()
plt.xlabel("patient's year of operation")
plt.ylabel('number of positive auxillary nodes')
plt.show()
plt.close()
sns.pairplot(hsd, hue = 'Sr_stat', vars = ['Age', 'Year', 'Aux_nodes'],  size = 4)
plt.show()