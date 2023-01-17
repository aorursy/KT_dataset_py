import os
os.getcwd()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
# Read datatset into framework
# Assigned column names as 

warnings.filterwarnings('ignore')
cancer_ds = pd.read_csv("haberman.csv", header = None, names = ['Age',
                          'Year_of_Operation', 'Auxil_Nodes_det', 'Survival_period'])

# Size of the dataset
print (cancer_ds.shape)
cancer_ds["Auxil_Nodes_det"].value_counts()
print(cancer_ds.head(5))
print(cancer_ds.columns)
print(cancer_ds.describe())
plt.close()
for idx, feature in enumerate(list(cancer_ds.columns)[0:3]):
    sns.FacetGrid(cancer_ds, hue="Survival_period", size=4).map(sns.distplot, feature).add_legend()
    plt.show()
count, bin_edges = np.histogram(cancer_ds['Age'], bins=10,density = True)
pdf = count / sum(count)
cdf = np.cumsum(pdf)
print("PDF of Age: ",pdf)
print(bin_edges)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel("Age")
cancer_ds.plot(kind = 'scatter', x = 'Age', y = 'Year_of_Operation')
plt.show()
# Multi-variate Analysis
sns.set_style("whitegrid");
sns.FacetGrid(cancer_ds, hue="Survival_period", size=4) \
   .map(plt.scatter, "Age", "Year_of_Operation") \
   .add_legend();
plt.show();
plt.close()
sns.set_style("whitegrid");
sns.pairplot(cancer_ds, hue="Survival_period", vars=['Age', 'Year_of_Operation', 'Auxil_Nodes_det'], size=5);
plt.show();
plt.figure(1)
plt.subplot(161)
sns.boxplot(x="Survival_period", y="Age",data = cancer_ds)
plt.subplot(163)
sns.boxplot(x="Survival_period", y="Year_of_Operation",data = cancer_ds)
plt.subplot(165)
sns.boxplot(x="Survival_period", y="Auxil_Nodes_det",data = cancer_ds)
plt.show()
sns.set_style('darkgrid')
plt.figure(1)
plt.subplot(161)
sns.violinplot(x='Survival_period',y='Age',data=cancer_ds)
#plt.show()
#Violin Plot using Patients operation year.
plt.subplot(163)
sns.violinplot(x='Survival_period',y='Year_of_Operation',data=cancer_ds)
#plt.show()
#Violin Plot using no. of positive axillary nodes.
plt.subplot(165)
sns.violinplot(x='Survival_period',y='Auxil_Nodes_det',data=cancer_ds)
plt.show()
sns.jointplot(x="Auxil_Nodes_det", y="Age", data=cancer_ds, kind="kde");
plt.show();
