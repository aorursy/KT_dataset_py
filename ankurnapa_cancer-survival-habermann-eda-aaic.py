import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("../input/habermans-survival-data-set/haberman.csv", names =["age","operation_year", "axilary_node","survival_status"])
print(df.shape)
print(df.columns)
df["survival_status"].value_counts()
df.plot(kind='scatter',x='axilary_node',y='age')
plt.grid()
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(df,hue="survival_status",size=8)\
.map(plt.scatter,"axilary_node","age","operation_year")\
.add_legend();
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")
plt.show();

plt.close();
sns.set_style("whitegrid");
plt.suptitle("Pair Plots")
sns.pairplot(df, hue="survival_status", height=3 , vars = ['age', 'operation_year', 'axilary_node'])


plt.show();

import numpy as np
df_long_survive = df.loc[df["survival_status"]==1];
df_short_survive = df.loc[df["survival_status"]==2];
plt.plot(df_long_survive["axilary_node"],np.zeros_like(df_long_survive['axilary_node']),'o')
plt.plot(df_short_survive["axilary_node"],np.zeros_like(df_short_survive['axilary_node']),'o')
plt.xlabel('Axillary Nodes')
plt.suptitle("1-D Scatter Plot", size=28)
plt.show()
sns.FacetGrid(df, hue="survival_status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.suptitle("PDF of Age",size=28)
plt.show();

sns.FacetGrid(df, hue="survival_status", size=6) \
   .map(sns.distplot, "operation_year") \
   .add_legend();
plt.suptitle("Historagms of Operation Year",size=28)
plt.show();
sns.FacetGrid(df, hue="survival_status", size=8) \
   .map(sns.distplot, "axilary_node") \
   .add_legend();
plt.suptitle("Historagms of Nodes",size=28)
plt.show();
count,bin_edges =np.histogram(df_long_survive['axilary_node'],bins=10, density=True)
pdf = count/(sum(count))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:],cdf)
plt.show()

count, bin_edges = np.histogram(df_short_survive['axilary_node'], bins=10, 
                                 density = True)
pdf = count/(sum(count))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();
print ("Median:-")
print(np.median(df_long_survive["axilary_node"]))
print(np.median(df_short_survive["axilary_node"]))

print("\n St.Dev")
print(np.std(df_long_survive["axilary_node"]))
print(np.std(df_short_survive["axilary_node"]))

print("\nQuantiles:")
print(np.percentile(df_long_survive["axilary_node"], np.arange(0, 100, 25)))
print(np.percentile(df_short_survive["axilary_node"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(df_long_survive["axilary_node"],90 ))
print(np.percentile(df_short_survive["axilary_node"],90))

from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(df_long_survive["axilary_node"]))
print(robust.mad(df_short_survive["axilary_node"]))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i,attr in enumerate(list(df.columns)[:-1]):
    sns.boxplot(x='survival_status',y=attr,data=df, ax=axes[i])
plt.suptitle("Box Plots",size=28)
plt.show()

    

sns.boxplot(x='survival_status',y='axilary_node', data=df)

    
plt.show()
plt.suptitle("Voilin Plots",size=28)
sns.violinplot(x="survival_status", y="axilary_node",data=df)
plt.legend

plt.show()
sns.violinplot(x="survival_status", y="age",data=df)
plt.legend
plt.show()
sns.violinplot(x="survival_status", y="operation_year",data=df)
plt.legend
plt.show()

sns.jointplot(x="age", y="axilary_node", data=df_long_survive, kind="kde");
plt.grid()
plt.show();
sns.jointplot(x="age", y="operation_year", data=df_long_survive, kind="kde");
plt.grid()
plt.show();