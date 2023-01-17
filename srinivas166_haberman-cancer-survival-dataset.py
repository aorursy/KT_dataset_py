import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



#read the data

#Information Regarding axillary nodes and how it leads to cancer "https://www.medicalnewstoday.com/articles/319713.php"

Data=pd.read_csv('../input/haberman.csv',names=['Age', 'Year', 'Axillary_nodes_dect', 'Surv_status'])

Data
print(Data.shape)#output shows no of (rows, columns)
print(Data.columns)#output shows the column names
Data['Surv_status'].value_counts()
#for age and year

Data.plot(kind='scatter', x='Age', y='Year')

plt.show()

#for age and year

sns.set_style('whitegrid')

sns.FacetGrid(Data,hue='Surv_status',height=4).map(plt.scatter,"Age","Year").add_legend()

plt.title("Scatter-plot of Age and year")

plt.show()
#for age and Axillary_nodes_dect

sns.set_style('whitegrid')

sns.FacetGrid(Data,hue='Surv_status',height=4) .map(plt.scatter,"Age","Axillary_nodes_dect").add_legend()

plt.title("Scatter-plot of Age and Axillary_nodes_dect")

plt.show()
#for year and Axillary_nodes_dect

sns.set_style('whitegrid')

sns.FacetGrid(Data,hue='Surv_status',height=4).map(plt.scatter,"Year","Axillary_nodes_dect").add_legend()

plt.title("Scatter-plot of year and Axillary_nodes_dect")

plt.show()


sns.set_style("whitegrid")

sns.pairplot(Data, hue="Surv_status", vars=['Age', 'Year', 'Axillary_nodes_dect'],height=4)

plt.show()
#distribution plot of age

sns.FacetGrid(Data, hue="Surv_status", height=5).map(sns.distplot, "Age").add_legend()

plt.title("Histogram of Age")

plt.show()

#distribution plot of year

sns.FacetGrid(Data, hue="Surv_status", height=5).map(sns.distplot, "Year").add_legend()

plt.title("Histogram of Year")

plt.show()
#distribution plot of Auxillary_nodes_dect

sns.FacetGrid(Data, hue="Surv_status", height=10) .map(sns.distplot, "Axillary_nodes_dect") .add_legend()

plt.title("Histogram of Axillary_nodes_dect")

plt.show()

surv_data = Data.loc[Data["Surv_status"] == 1]

counts, bin_edges = np.histogram(surv_data['Axillary_nodes_dect'], bins=10,density = True)

pdf = counts/(sum(counts))



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.xlabel("Axillary_nodes_dect")

plt.legend(['PDF', 'CDF'])

plt.title("PDF and CDF of Axillary_nodes_detected")



dead_data = Data.loc[Data["Surv_status"] == 2]



counts, bin_edges = np.histogram(dead_data['Axillary_nodes_dect'], bins=10,density = True)

pdf = counts/(sum(counts))



cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.xlabel("Axillary_nodes_dect")

plt.legend(['PDF', 'CDF'])

sns.boxplot(x='Surv_status',y='Axillary_nodes_dect',hue = "Surv_status", data=Data).set_title("BOX-plot for survival_status and axillary_nodes_dect")

plt.show()
sns.violinplot(x='Surv_status',y='Axillary_nodes_dect',hue = "Surv_status", data=Data, size=10).set_title("violin plot for Surv_status and axillary_nodes_dect")

plt.show()
sns.jointplot(x='Surv_status',y='Axillary_nodes_dect',data=Data, kind="kde")

plt.show()