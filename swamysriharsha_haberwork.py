#importing required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Loading data set
hab = pd.read_csv('../input/haberman.csv') #here, hab acts as a pandas data frame
print(hab.shape) #It gives total number of rows and columns (i.e; data points and features)
print(hab.head()) #it gives the results of top most rows from the data set to observe the structure of it
colnames = ["age", "year", "nof_nodes", "status"] #adds column names to it
hab = pd.read_csv('../input/haberman.csv', names=colnames)
print(hab.head()) #it gives the results of top most rows from the data set to observe the structure of it
print(hab.shape) #It gives total number of rows and columns (i.e; data points and features)
print(hab.columns) #It gives all the column names in the data set
#to find how many data points for each class are present (or)
#to find how many patients survived 5 years or longer (1), how many patients died within 5 year (2)
hab['status'].value_counts()
hab.plot(kind="scatter", x="age", y="nof_nodes");
plt.show()
# here I have considered age as an x-axis and nof_nodes as an y-axis
sns.set_style("whitegrid");
sns.FacetGrid(hab, hue="status", size=5) \
   .map(plt.scatter, "age", "nof_nodes") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.pairplot(hab, hue="status", size=3);
plt.show()
sns.FacetGrid(hab, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
# Here we have made a distribution plot which consists of both histogram and PDF in it
sns.FacetGrid(hab, hue="status", size=8) \
   .map(sns.distplot, "nof_nodes") \
   .add_legend();
plt.show();
# Here we have made a distribution plot which consists of both histogram and PDF in it
sns.FacetGrid(hab, hue="status", size=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.show();
# Here we have made a distribution plot which consists of both histogram and PDF in it
hab_status1 = hab.loc[hab["status"] == 1];
hab_status2 = hab.loc[hab["status"] == 2];
print(hab_status1.count())
print(hab_status2.count())
hab_status1.describe()
hab_status2.describe()
sns.boxplot(x='status',y='nof_nodes', data=hab)
plt.show()
sns.violinplot(x="status", y="nof_nodes", data=hab, size=8)
plt.show()
print("\nMedians:")
print(np.median(hab_status1["nof_nodes"]))
print(np.median(hab_status2["nof_nodes"]))

print("\nQuantiles:")
print(np.percentile(hab_status1["nof_nodes"],np.arange(0, 100, 25)))
print(np.percentile(hab_status2["nof_nodes"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(hab_status1["nof_nodes"],90))
print(np.percentile(hab_status2["nof_nodes"],90))     

