import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os
#Load haberman.csv into a pandas dataFrame
feaures_name = ['age','year','axil_nodes','Surv_status']
df = pd.read_csv('../input/haberman.csv',names=feaures_name)

#Verify the data that has been loaded
df.head(12)
#print the unique values of the target column

print(df['Surv_status'].unique())
df.Surv_status.replace([1, 2], ['Survived', 'Not Survived'], inplace = True)
df.head(12)

# number of data-points and features?
print (df.shape)
#data points for each class (label)
print(df['Surv_status'].value_counts())
#scatter plot for age and year
sb.set_style("whitegrid");
sb.FacetGrid(df, hue="Surv_status", size=4) \
   .map(plt.scatter, "age", "year") \
   .add_legend();
plt.show();
#scatter plot for age and axil_nodes
sb.set_style("whitegrid");
sb.FacetGrid(df, hue="Surv_status", size=4) \
   .map(plt.scatter, "age", "axil_nodes") \
   .add_legend();
plt.show();
#scatter plot for year and axil_nodes
sb.set_style("whitegrid");
sb.FacetGrid(df, hue="Surv_status", size=4) \
   .map(plt.scatter, "year", "axil_nodes") \
   .add_legend();
plt.show();
#pair plot
plt.close();
sb.set_style("whitegrid");
sb.pairplot(df, hue="Surv_status", size=3);
plt.show()
#Probability Density Functions (PDF) for age
survived = df.loc[df["Surv_status"] == "Survived"];
not_survived = df.loc[df["Surv_status"] == "Not Survived"];

sb.FacetGrid(df, hue="Surv_status", size=5) \
   .map(sb.distplot, "age") \
   .add_legend();
plt.show();
#Probability Density Functions (PDF) for year
sb.FacetGrid(df, hue="Surv_status", size=5) \
   .map(sb.distplot, "year") \
   .add_legend();
plt.show();
#Probability Density Functions (PDF) for axil_nodes 
sb.FacetGrid(df, hue="Surv_status", size=5) \
   .map(sb.distplot, "axil_nodes") \
   .add_legend();
plt.show();
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:], cdf,label='survived cdf')
plt.legend()

counts, bin_edges = np.histogram(not_survived['age'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not survived pdf')
plt.plot(bin_edges[1:], cdf,label='not survived cdf')
plt.legend()

plt.xlabel("age")
plt.show()
counts, bin_edges = np.histogram(survived['year'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:], cdf,label='survived cdf')
plt.legend()

counts, bin_edges = np.histogram(not_survived['year'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not survived pdf')
plt.plot(bin_edges[1:], cdf,label='not survived cdf')
plt.legend()

plt.xlabel("year")
plt.show()
counts, bin_edges = np.histogram(survived['axil_nodes'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='survived pdf')
plt.plot(bin_edges[1:], cdf,label='survived cdf')
plt.legend()

counts, bin_edges = np.histogram(not_survived['axil_nodes'], bins=10, 
                                 density = True)

pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='not survived pdf')
plt.plot(bin_edges[1:], cdf,label='not survived cdf')
plt.legend()

plt.xlabel("axil_nodes")
plt.show()
sb.boxplot(x='Surv_status',y='age', data=df)
plt.show()
sb.boxplot(x='Surv_status',y='year', data=df)
plt.show()
sb.boxplot(x='Surv_status',y='axil_nodes', data=df)
plt.show()
sb.violinplot(x="Surv_status", y="age", data=df, size=8)
plt.show()
sb.violinplot(x="Surv_status", y="year", data=df, size=8)
plt.show()
sb.violinplot(x="Surv_status", y="axil_nodes", data=df, size=8)
plt.show()