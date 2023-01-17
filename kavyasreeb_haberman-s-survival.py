import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
print(os.listdir('../input')) #checking input dataset
haberman_df = pd.read_csv('../input/haberman.csv/haberman.csv')

haberman_df.head()
print (haberman_df.shape)   #shows datapoints and features                     
print (haberman_df.columns) #displays column names in our dataset
haberman_df["status"].value_counts()

print(list(haberman_df['status'].unique())) # print the unique values of the target column(status)
haberman_df['status'] = haberman_df['status'].map({1:'YES', 2:'NO'}) #mapping the value '1' to 'YES'and value '2' to 'NO'
haberman_df.head() #printing the first 5 records from the dataset.
one = haberman_df.loc[haberman_df["status"] == "YES"]
two = haberman_df.loc[haberman_df["status"] == "NO"]
plt.plot(one["age"], np.zeros_like(one["age"]), 'o',label='YES')
plt.plot(two["age"], np.zeros_like(two["age"]), 'o',label='NO')
plt.title("1-D scatter plot for age")
plt.xlabel("age")
plt.legend(title="survival_status")
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(haberman_df, hue="status", height=6) \
   .map(plt.scatter, "age", "nodes") \
   .add_legend();
plt.show();

sns.set_style("whitegrid")
sns.pairplot(haberman_df, diag_kind="kde", hue="status", height=4)
plt.show()

sns.FacetGrid(haberman_df, hue="status", height=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.title("PDF of age")
plt.show();
sns.FacetGrid(haberman_df, hue="status", height=5) \
   .map(sns.distplot, "year") \
   .add_legend();
plt.title("PDF of year")
plt.show();
sns.FacetGrid(haberman_df, hue="status", height=5) \
   .map(sns.distplot, "nodes") \
   .add_legend();
plt.title("PDF of nodes")
plt.show();
# the patient survived 5 years or longer
counts, bin_edges = np.histogram(one['nodes'], bins=10, density = True)
pdf1 = counts/(sum(counts))
print(pdf1);
print(bin_edges)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges[1:],pdf1)
plt.plot(bin_edges[1:], cdf1)
 
# the patient dies within 5 years
counts, bin_edges = np.histogram(two['nodes'], bins=10, density = True)
pdf2 = counts/(sum(counts))
print(pdf2)
print(bin_edges)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges[1:],pdf2)
plt.plot(bin_edges[1:], cdf2)

label = ["pdf of patient_survived", "cdf of patient_survived", "pdf of patient_died", "cdf of patient_died"]
plt.legend(label)
plt.xlabel("positive_lymph_node")
plt.title("pdf and cdf for positive_lymph_node")
plt.show();
# the patient survived 5 years or longer
counts, bin_edges = np.histogram(one['age'], bins=10, density = True)
pdf1 = counts/(sum(counts))
print(pdf1);
print(bin_edges)
cdf1 = np.cumsum(pdf1)
plt.plot(bin_edges[1:],pdf1)
plt.plot(bin_edges[1:], cdf1)
 
# the patient dies within 5 years
counts, bin_edges = np.histogram(two['age'], bins=10, density = True)
pdf2 = counts/(sum(counts))
print(pdf2)
print(bin_edges)
cdf2 = np.cumsum(pdf2)
plt.plot(bin_edges[1:],pdf2)
plt.plot(bin_edges[1:], cdf2)

label = ["pdf of patient_survived", "cdf of patient_survived", "pdf of patient_died", "cdf of patient_died"]
plt.legend(label)
plt.xlabel("age")
plt.title("pdf and cdf for age")
plt.show();
sns.boxplot(x='status',y='age', data=haberman_df)
plt.title("Box_plot for age and survival status")
plt.show()

sns.boxplot(x='status',y='year', data=haberman_df)
plt.title("Box_plot for year and survival status")
plt.show()

sns.boxplot(x='status',y='nodes', data=haberman_df)
plt.title("Box_plot for nodes and survival status")
plt.show()
sns.violinplot(x="status", y="age", data=haberman_df, size=8)
plt.title("Violin plot for age and survival status")
plt.show()

sns.violinplot(x="status", y="year", data=haberman_df, size=8)
plt.title("Violin plot for year and survival status")
plt.show()

sns.violinplot(x="status", y="nodes", data=haberman_df, size=8)
plt.title("Violin plot for nodes and survival status")
plt.show()
sns.jointplot(x="age", y="year", data=haberman_df, kind="kde");
plt.show();