import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset and add headers to the columns as described in the dataset.
haberman_df = pd.read_csv('../input/haberman.csv', names=['Age', 'Operation Year', 'Positive Axiliary Nodes', 'Survival Status After 5 Years'])
haberman_df.head()
# Map Numeric value of 'Survival Status After 5 Years' to values - "Survived", "Died"
survival_status_dict = {1:"Survived",2:"Died"}
haberman_df['Survival Status After 5 Years'] = haberman_df['Survival Status After 5 Years'].map(survival_status_dict)
haberman_df.head()
haberman_df.info()
haberman_df.describe()
haberman_df.shape
haberman_df['Survival Status After 5 Years'].value_counts()
haberman_survived_df = haberman_df.loc[haberman_df['Survival Status After 5 Years'] == 'Survived']
haberman_died_df = haberman_df.loc[haberman_df['Survival Status After 5 Years'] == 'Died']

plt.plot(haberman_survived_df['Age'], np.zeros_like(haberman_survived_df['Age']), 'o')
plt.plot(haberman_died_df['Age'], np.zeros_like(haberman_died_df['Age']), 'o')
plt.show()
# Distribution Plot
# 1.1
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5) \
    .map(sns.distplot, "Age") \
    .add_legend();
plt.show();
plt.close()
plt.plot(haberman_survived_df['Operation Year'], np.zeros_like(haberman_survived_df['Operation Year']), 'o')
plt.plot(haberman_died_df['Operation Year'], np.zeros_like(haberman_died_df['Operation Year']), 'o')
plt.show()
# 1.2
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5) \
    .map(sns.distplot, "Operation Year") \
    .add_legend();
plt.show();
plt.close()
plt.plot(haberman_survived_df['Positive Axiliary Nodes'], np.zeros_like(haberman_survived_df['Positive Axiliary Nodes']), 'o')
plt.plot(haberman_died_df['Positive Axiliary Nodes'], np.zeros_like(haberman_died_df['Positive Axiliary Nodes']), 'o')
plt.show()
# 1.3
sns.FacetGrid(haberman_df, hue="Survival Status After 5 Years", size=5) \
    .map(sns.distplot, "Positive Axiliary Nodes") \
    .add_legend();
plt.show();
#1.4
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Age of Patients")
plt.xlabel("Age")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();
# 1.5
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Operation Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Operation Years of Patients")
plt.xlabel("Operation Year")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();
# 1.6
# PDF, CDF as per Patient's Age

# Survived Patients
counts, bin_edges = np.histogram(haberman_survived_df['Positive Axiliary Nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

# Died Patients
counts, bin_edges = np.histogram(haberman_died_df['Positive Axiliary Nodes'], bins=10,
                                 density = True)
pdf = counts/(sum(counts))
# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("PDF and CDF for Positive Axiliary Nodes of Patients")
plt.xlabel("Positive Axiliary Nodes")
plt.ylabel("% of Patient's")

label = ["PDF (Survived)", "CDF (Survived)", "PDF (Died)", "CDF (Died)"]
plt.legend(label)

plt.show();
# 1.7
sns.boxplot(x='Survival Status After 5 Years',y='Age', data=haberman_df)
plt.show()
# 1.8
sns.boxplot(x='Survival Status After 5 Years',y='Operation Year', data=haberman_df)
plt.show()
# 1.9
sns.boxplot(x='Survival Status After 5 Years',y='Positive Axiliary Nodes', data=haberman_df)
plt.show()
# 1.10
sns.violinplot(x='Survival Status After 5 Years',y='Age', data=haberman_df, size=8)
plt.show()
# 1.11
sns.violinplot(x='Survival Status After 5 Years',y='Operation Year', data=haberman_df, size=8)
plt.show()
# 1.12
sns.violinplot(x='Survival Status After 5 Years',y='Positive Axiliary Nodes', data=haberman_df, size=8)
plt.show()
# 2.1
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5) \
   .map(plt.scatter, 'Age', 'Operation Year') \
   .add_legend();
plt.show();
# 2.2
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5) \
   .map(plt.scatter, 'Age', 'Positive Axiliary Nodes') \
   .add_legend();
plt.show();
# 2.3
sns.set_style('whitegrid');
sns.FacetGrid(haberman_df, hue='Survival Status After 5 Years', size=5) \
   .map(plt.scatter, 'Operation Year', 'Positive Axiliary Nodes') \
   .add_legend();
plt.show();
# 2.4
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman_df, hue="Survival Status After 5 Years", size=5);
plt.show()
# 2.5
sns.jointplot(x="Age", y="Positive Axiliary Nodes", data=haberman_survived_df, kind="kde");
plt.show();
# 2.6
sns.jointplot(x="Operation Year", y="Positive Axiliary Nodes", data=haberman_survived_df, kind="kde");
plt.show();
# 2.7
sns.jointplot(x="Age", y="Operation Year", data=haberman_survived_df, kind="kde");
plt.show();
