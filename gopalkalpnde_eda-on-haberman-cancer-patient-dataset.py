import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#Load haberman.csv into a pandas dataFrame.
df = pd.read_csv("../input/haberman.csv")


# (Q) how many data-points and features?
print (df.shape)
#(Q) What are the column names in our dataset?
print (df.columns)

# naming the columns
df.columns = ['age','operation_year','axil_nodes','surv_status']
print(df.columns)
print(df.shape)
# How many data points for each class are present? 

df["surv_status"].value_counts()
# Distribution of axil_nodes
sns.FacetGrid(df, hue="surv_status", size=5) \
   .map(sns.distplot, "axil_nodes") \
   .add_legend();
plt.show();
print(df['axil_nodes'].mean())
# Distribution of axil_nodes
sns.FacetGrid(df, hue="surv_status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
# Distribution of axil_nodes
sns.set_style('whitegrid')
sns.FacetGrid(df, hue="surv_status", size=5) .map(sns.distplot, "operation_year").add_legend();
plt.show();
# Dividing data on the basis of classes

survived = df.loc[df["surv_status"] == 1];
died = df.loc[df["surv_status"] == 2];
# Plots of CDF of axil_nodes for two classes.

# Survived 5 years or longer 
counts, bin_edges = np.histogram(survived['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of axil_nodes which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf, label = "CDF of axil_nodes which Survived 5 years or longer ")


# Died within 5 years of operation
counts, bin_edges = np.histogram(died['axil_nodes'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of axil_nodes which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf, label = "PDF of axil_nodes which Died within 5 years of operation")
plt.legend(loc = 'best')
plt.xlabel("axil_nodes")

plt.show();
# Plots of CDF of petal_length for various types of flowers.

# Misclassification error if you use 
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of age which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf,label = "CDF of age which Survived 5 years or longer ")


counts, bin_edges = np.histogram(died['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label = "PDF of age which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf,label = "PDF of age which Died within 5 years of operation")
plt.legend(loc='best')
plt.xlabel("age")


plt.show();
# Plots of CDF of operation_year for various types.

counts, bin_edges = np.histogram(survived['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of operation_year which Survived 5 years or longer ")
plt.plot(bin_edges[1:], cdf, label = "CDF of operation_year which Survived 5 years or longer ")



counts, bin_edges = np.histogram(died['operation_year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf, label = "PDF of operation_year which Died within 5 years of operation")
plt.plot(bin_edges[1:], cdf, label = "CDF of operation_year which Died within 5 years of operation")
plt.xlabel("operation_year")
plt.legend(loc = 'best')

plt.show();
#Mean, Std-deviation of age
print("Means:")
print(np.mean(survived["age"]))
print(np.mean(died["age"]))

print("\nStd-dev:");
print(np.std(survived["age"]))
print(np.std(died["age"]))
#Mean, Std-deviation of axil_nodes
print("Means:")
print(np.mean(survived["axil_nodes"]))
print(np.mean(died["axil_nodes"]))

print("\nStd-dev:");
print(np.std(survived["axil_nodes"]))
print(np.std(died["axil_nodes"]))
#Mean, Std-deviation of operation_year
print("Means:")
print(np.mean(survived["operation_year"]))
print(np.mean(died["operation_year"]))

print("\nStd-dev:");
print(np.std(survived["operation_year"]))
print(np.std(died["operation_year"]))
# box plot for axil_nodes
sns.boxplot(x='surv_status',y='axil_nodes', data=df)
plt.show()
# box plot for age
sns.boxplot(x='surv_status',y='age', data=df)
plt.show()
# box plot for operation_year
sns.boxplot(x='surv_status',y='operation_year', data=df)
plt.show()
# Violin plot for Axil_nodes
sns.violinplot(x='surv_status',y='axil_nodes', data=df, size=8)
plt.show()
# Violin plot for age
sns.violinplot(x='surv_status',y='age', data=df, size=8)
plt.show()
# Violin plot for Axil_nodes
sns.violinplot(x='surv_status',y='operation_year', data=df, size=8)
plt.show()
# 2-D Scatter plot with color-coding for each class i.e.  
#    1 = the patient survived 5 years or longer 
#    2 = the patient died within 5 years

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4) \
   .map(plt.scatter, "age", "axil_nodes") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4) \
   .map(plt.scatter, "age", "operation_year") \
   .add_legend();
plt.show();

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="surv_status", size=4) \
   .map(plt.scatter, "axil_nodes", "operation_year") \
   .add_legend();
plt.show();

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

    
xs = df["age"]
ys = df["operation_year"]
zs = df["axil_nodes"]    

ax.scatter(xs, ys, zs)

ax.set_xlabel('age')
ax.set_ylabel('operation_year')
ax.set_zlabel('axil_nodes')

plt.show()
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="surv_status", vars = ['age','operation_year','axil_nodes'],size=5);
plt.show()
#1-D scatter plot of axil_nodes
plt.plot(survived["axil_nodes"], np.zeros_like(survived['axil_nodes']), 'o')
plt.plot(died["axil_nodes"], np.zeros_like(died['axil_nodes']), '^')

plt.show()
#1-D scatter plot of operation_year
plt.plot(survived["operation_year"], np.zeros_like(survived['operation_year']), 'o')
plt.plot(died["operation_year"], np.zeros_like(died['operation_year']), '^')

plt.show()
#1-D scatter plot of age
plt.plot(survived["age"], np.zeros_like(survived['age']), 'o')
plt.plot(died["age"], np.zeros_like(died['age']), '^')

plt.show()