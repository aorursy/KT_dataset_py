import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

hdata = pd.read_csv("../input/haberman.csv",header=None ,names = ["Age", "Operation_year", "Axil_nodes", "Surv_status"])

hdata['Surv_status']=hdata['Surv_status'].map({2:"NO",1:'YES'})

hdata.head()
print (hdata.shape)
print (hdata.columns)
sns.set_style("whitegrid");

sns.FacetGrid(hdata,hue="Surv_status",height=5).map(plt.scatter, "Surv_status", "Axil_nodes").add_legend();

plt.show();

#https://seaborn.pydata.org/tutorial/categorical.html
#Univariate analysis

patient_survived = hdata.loc[hdata["Surv_status"] == 'YES'];

patient_died = hdata.loc[hdata["Surv_status"] == 'NO'];

plt.plot(patient_survived["Axil_nodes"], np.zeros_like(patient_survived['Axil_nodes']), 'o' )

plt.plot(patient_died["Axil_nodes"], np.zeros_like(patient_died['Axil_nodes']), 'o')

plt.title("1-D scatter plot for Axil_nodes")

plt.show()
sns.set_style("whitegrid");

sns.FacetGrid(hdata,hue="Surv_status",height=5).map(plt.scatter,  "Operation_year" , "Age").add_legend();    

plt.show();
sns.set_style("whitegrid");

sns.FacetGrid(hdata,hue="Surv_status",height=5).map(plt.scatter, "Age" , "Axil_nodes").add_legend();

plt.show();

sns.set_style("whitegrid");

sns.pairplot(hdata, hue="Surv_status", height=4);

plt.show()
sns.FacetGrid(hdata,hue='Surv_status',height=5).map(sns.distplot,'Age').add_legend()

plt.show()
sns.FacetGrid(hdata,hue='Surv_status',height=5).map(sns.distplot,"Operation_year").add_legend()

plt.show()
sns.FacetGrid(hdata,hue='Surv_status',height=5).map(sns.distplot,"Axil_nodes").add_legend()

plt.show()
counts, bin_edges = np.histogram(patient_survived['Axil_nodes'], bins=15, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])



counts, bin_edges = np.histogram(patient_died['Axil_nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])

plt.show();







counts, bin_edges = np.histogram(patient_survived['Age'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])



counts, bin_edges = np.histogram(patient_died['Age'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])

plt.show();





counts, bin_edges = np.histogram(patient_survived['Operation_year'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges);

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])



counts, bin_edges = np.histogram(patient_died['Operation_year'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:], cdf)

plt.legend('surv_status')

plt.legend(['Survived_PDF', 'Survived_CDF','Died_PDF', 'Died_CDF'])

plt.show();

#appliedaicourse.com
sns.boxplot(x='Surv_status',y='Age', data=hdata)

plt.show()



sns.boxplot(x='Surv_status',y='Operation_year', data=hdata)

plt.show()



sns.boxplot(x='Surv_status',y='Axil_nodes', data=hdata)

plt.show()
sns.violinplot(x='Surv_status',y='Age', data=hdata)

plt.show()



sns.violinplot(x='Surv_status',y='Operation_year', data=hdata)

plt.show()



sns.violinplot(x='Surv_status',y='Axil_nodes', data=hdata)

plt.show()
print("Summary Statistics of Patients")

hdata.describe()
print("Summary Statistics of Patient who Survived.")

patient_survived.describe()
print("Summary Statistics of Patient who Not Survived.")

patient_died.describe()