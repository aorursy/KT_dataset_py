import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cols = ['Age', 'Operation_year', 'Axil_nodes', 'Survival_status']
hb = pd.read_csv('../input/haberman.csv', names = cols)

# Input data files are available in the "../input/" directory.
print(hb.shape)
print(hb.columns)
hb["Survival_status"].value_counts()
#statistical parameters
hb.describe()
#Bivariate Analysis using scatter plots 
sns.set_style("whitegrid");
sns.FacetGrid(hb, hue="Survival_status", size=4) \
   .map(plt.scatter, "Axil_nodes", "Operation_year") \
   .add_legend();
plt.show();
#Bivariate Analysis using Pairwise plots 
plt.close();
sns.set_style("whitegrid");
sns.pairplot(hb, hue="Survival_status", size=3);
plt.show()
#Univariate analysis
patient_survived = hb.loc[hb["Survival_status"] == 1];
patient_died = hb.loc[hb["Survival_status"] == 2];
plt.plot(patient_survived["Axil_nodes"], np.zeros_like(patient_survived['Axil_nodes']), 'o')
plt.plot(patient_died["Axil_nodes"], np.zeros_like(patient_died['Axil_nodes']), 'o')
plt.show()
sns.FacetGrid(hb, hue="Survival_status", size=5) \
   .map(sns.distplot, "Axil_nodes") \
   .add_legend();
plt.show();
sns.FacetGrid(hb, hue="Survival_status", size=5) \
   .map(sns.distplot, "Age") \
   .add_legend();
plt.show();
sns.FacetGrid(hb, hue="Survival_status", size=5) \
   .map(sns.distplot, "Operation_year") \
   .add_legend();
plt.show();
##Univariate analysis using PDF and CDF
counts, bin_edges = np.histogram(patient_survived['Axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)


counts, bin_edges = np.histogram(patient_died['Axil_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();
#Mean, Variance, Std-deviation,  
print("Means:")
print(np.mean(patient_survived["Axil_nodes"]))
print(np.mean(patient_died["Axil_nodes"]))
print("\nStd-dev:");
print(np.std(patient_survived["Axil_nodes"]))
print(np.std(patient_died["Axil_nodes"]))

#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(patient_survived["Axil_nodes"]))
print(np.median(patient_died["Axil_nodes"]))

print("\nQuantiles:")
print(np.percentile(patient_survived["Axil_nodes"],np.arange(0,100,25)))
print(np.percentile(patient_died["Axil_nodes"],np.arange(0,100,25)))

print("\n90th Percentiles:")
print(np.percentile(patient_survived["Axil_nodes"],90))
print(np.percentile(patient_died["Axil_nodes"],90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(patient_survived["Axil_nodes"]))
print(robust.mad(patient_died["Axil_nodes"]))
#Univariate analysis using Boxplot
sns.boxplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.boxplot(x='Survival_status',y='Age', data=hb)
plt.show()
sns.boxplot(x='Survival_status',y='Operation_year', data=hb)
plt.show()
#Univariate analysis using Violinplot
sns.violinplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.violinplot(x='Survival_status',y='Axil_nodes', data=hb)
plt.show()
sns.violinplot(x='Survival_status',y='Age', data=hb)
plt.show()
sns.jointplot(x="Operation_year", y="Age", data=hb, kind="kde");
plt.show();
sns.jointplot(x="Axil_nodes", y="Age", data=hb, kind="kde");
plt.show();
