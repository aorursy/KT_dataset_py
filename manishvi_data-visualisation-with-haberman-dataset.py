import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load haberman.csv into a pandas dataFrame."
haberman = pd.read_csv("../input/haberman.csv")
print(haberman.shape)
haberman.columns = ["Age", "Operation_Year", "Axillary_Nodes", "Survival_Status"]
haberman.columns
haberman.head()
# total number of classifications for predictor 
haberman["Survival_Status"].value_counts()
#Checking datatypes of Feature
haberman.dtypes
# Check is there any null values
haberman.isnull().values.any()
# Scatter plot for Age against Year of operation
haberman.plot(kind='scatter', x='Age', y='Operation_Year') ;
plt.show()

# It is very hard to make any sense out these points.
# let's assign color code to data points belonginig to respective feature.
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=6) \
   .map(plt.scatter, "Age", "Operation_Year") \
   .add_legend();
plt.show();


# Scatter plot for Age against Axillary nodes
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Survival_Status", size=6) \
   .map(plt.scatter, "Age", "Axillary_Nodes") \
   .add_legend();
plt.show();
# pairwise scatter plot: Pair-Plot
# plt.close();
# sns.set_style("whitegrid");
# sns.pairplot(haberman, hue="Survival_Status", size=3);
# plt.show()

# NOTE: the diagnol elements are PDFs for each feature.
# Plotting pairplot for Class variable 'Sirvival status' with "Age", "Year of operation", "axillary nodes" as x and y axises
plt.close()
sns.pairplot(haberman, hue="Survival_Status", vars=["Age", "Operation_Year", "Axillary_Nodes"], size=3)
plt.show()
#3D scattered plot "Age", "Year of operation", "axillary nodes", "Survival status"
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')

x=haberman["Age"]
y=haberman["Operation_Year"]
z=haberman["Axillary_Nodes"]

ax.scatter(x,y,z,marker='o', c='r');

ax.set_xlabel('Age')
ax.set_ylabel('Operation_year')
ax.set_zlabel('Axillary_Nodes')

plt.show()
#showing datasets using HISTOGRAM
import numpy as np
Survived = haberman.loc[haberman["Survival_Status"] == 1];
Non_Survived = haberman.loc[haberman["Survival_Status"] == 2];

plt.plot(Survived["Age"], np.zeros_like(Survived['Age']), 'o')
plt.plot(Non_Survived["Age"], np.zeros_like(Non_Survived['Age']), 'o')


plt.show()
# Perform univariate analysis 
def plotFeature(feature_var, class_var):
    sns.set_style(style="whitegrid")
    sns.FacetGrid(data=haberman, hue=class_var, size=5) \
    .map(sns.distplot, feature_var) \
    .add_legend()
plotFeature("Age", "Survival_Status")
plt.show()
plotFeature("Operation_Year", "Survival_Status")
plt.show()
plotFeature("Axillary_Nodes", "Survival_Status")
plt.show()
#Get the counts and bin edges of axillary nodes whose survival status is 1
counts, bin_edges = np.histogram(Survived["Axillary_Nodes"], bins=30, density=True)
print (counts)
print (bin_edges)

#Get the counts and bin edges of axillary nodes whose survival status is 2
counts_Non, bin_edges_Non = np.histogram(Non_Survived["Axillary_Nodes"], bins=30, density=True)
print (counts_Non)
print (bin_edges_Non)

#PDF and CDF of survived
pdf_axillary_survived = counts/sum(counts)
cdf_axillary_survived = np.cumsum(pdf_axillary_survived)

#PDFand CDF of non survived
pdf_axillary_Non_survived = counts_Non/sum(counts_Non)
cdf_axillary_Non_survived = np.cumsum(pdf_axillary_Non_survived)

# Plot survived and non survived PDF, CDF in a single plot
plt.plot(bin_edges[1:], pdf_axillary_survived)
plt.plot(bin_edges[1:], cdf_axillary_survived)
plt.plot(bin_edges_Non[1:], pdf_axillary_Non_survived)
plt.plot(bin_edges_Non[1:], cdf_axillary_Non_survived)
#plt.xticks(np.linspace(0,50,13))
plt.xlabel("Axillary Node")

plt.legend(["Survived more than 5 years PDF", "Survived more than 5 years CDF", "Not survived more than 5 years PDF", "Not Survived more than 5 years CDF" ])
plt.show()

#Get the counts and bin edges of axillary nodes whose survival status is 1
counts, bin_edges = np.histogram(Survived["Operation_Year"], bins=30, density=True)
print (counts)
print (bin_edges)

#Get the counts and bin edges of axillary nodes whose survival status is 2
counts_Non, bin_edges_Non = np.histogram(Non_Survived["Operation_Year"], bins=30, density=True)
print (counts_Non)
print (bin_edges_Non)

#PDF and CDF of survived
pdf_axillary_survived = counts/sum(counts)
cdf_axillary_survived = np.cumsum(pdf_axillary_survived)

#PDFand CDF of non survived
pdf_axillary_Non_survived = counts_Non/sum(counts_Non)
cdf_axillary_Non_survived = np.cumsum(pdf_axillary_Non_survived)

# Plot survived and non survived PDF, CDF in a single plot
plt.plot(bin_edges[1:], pdf_axillary_survived)
plt.plot(bin_edges[1:], cdf_axillary_survived)
plt.plot(bin_edges_Non[1:], pdf_axillary_Non_survived)
plt.plot(bin_edges_Non[1:], cdf_axillary_Non_survived)
#plt.xticks(np.linspace(0,50,13))
plt.xlabel("Year of operation")

plt.legend(["Survived more than 5 years PDF", "Survived more than 5 years CDF", "Not survived more than 5 years PDF", "Not Survived more than 5 years CDF" ])
plt.show()
#Mean with an outlier.
print(np.mean(np.append(Survived["Axillary_Nodes"],550)));

print(np.mean(Survived["Axillary_Nodes"]))
print(np.mean(Non_Survived["Axillary_Nodes"]))

print("\nStd-dev:");
print(np.std(Survived["Axillary_Nodes"]))
print(np.std(Non_Survived["Axillary_Nodes"]))

#Median, Quantiles, Percentiles, IQR.
print("\nMedians:")
print(np.median(Survived["Axillary_Nodes"]))
#Median with an outlier
print(np.median(np.append(Survived["Axillary_Nodes"],550)));
print(np.median(Non_Survived["Axillary_Nodes"]))


print("\nQuantiles:")
print(np.percentile(Survived["Axillary_Nodes"],np.arange(0, 100, 25)))
print(np.percentile(Non_Survived["Axillary_Nodes"],np.arange(0, 100, 25)))


print("\n90th Percentiles:")
print(np.percentile(Survived["Axillary_Nodes"],90))
print(np.percentile(Non_Survived["Axillary_Nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(Survived["Axillary_Nodes"]))
print(robust.mad(Non_Survived["Axillary_Nodes"]))
#Box Plots w.r.t Age_Of_Patient
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Age",data=haberman)


#Box Plots w.r.t Operation_Year
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Operation_Year",data=haberman)

#Box Plots w.r.t Axil_nodes
plt.figure(figsize=(8,5))
sns.boxplot(x='Survival_Status',y="Axillary_Nodes",data=haberman)
plt.legend()
plt.show()
#Violin plots w.r.t Age_Of_Patient
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Age",data=haberman)

#Violin plots w.r.t Operation_Year
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Operation_Year",data=haberman)

#Violin plots w.r.t Axil_nodes
plt.figure(figsize=(8,5))
sns.violinplot(x='Survival_Status',y="Axillary_Nodes",data=haberman)
plt.show()
# Multivariate probability density
sns.jointplot(x= 'Age',kind = 'kde', y='Operation_Year', data =haberman)
plt.show()
sns.jointplot(x= 'Age',kind = 'kde', y='Axillary_Nodes', data =haberman)
plt.show()
sns.jointplot(x= 'Operation_Year',kind = 'kde', y='Axillary_Nodes', data =haberman)
plt.show()