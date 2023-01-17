import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df2 = pd.read_csv('../input/habermans-survival-data-set/haberman.csv',header=None,names=['age','year','nodes','status'])
print(df2)
df2.shape
df2.columns
df2["status"].value_counts()

print(df2.info())

sns.FacetGrid(df2, hue="status", size=5).map(sns.distplot, "age").add_legend()
plt.title('Histogram & PDF of age with respect to status of 5 year survival or not')
plt.show()


sns.FacetGrid(df2, hue="status", size=5).map(sns.distplot, "year").add_legend();
plt.title('Histogram & PDF of year with respect to status of 5 year survival or not')

plt.show();
sns.FacetGrid(df2, hue="status", size=5).map(sns.distplot, "nodes").add_legend();
plt.title('Histogram & PDF of nodes with respect to status of 5 year survival or not')

plt.show();
counts, bin_edges = np.histogram(df2['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:], cdf,label='cdf')
plt.xlabel("Age")
plt.ylabel("Probability")
plt.legend()
plt.title('PDF & CDF for AGE')


plt.show();
counts, bin_edges = np.histogram(df2['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:], cdf,label='cdf')
plt.xlabel("year")
plt.ylabel("Probability")
plt.legend()
plt.title('PDF & CDF for YEAR')





plt.show();
counts, bin_edges = np.histogram(df2['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,label='pdf')
plt.plot(bin_edges[1:], cdf,label='cdf')
plt.xlabel("nodes")
plt.ylabel("Probability")
plt.legend()
plt.title('PDF & CDF for no. of year of treatment')




plt.show();
sns.catplot(x="age", kind="box", palette="ch:.25", data=df2,orient='v');
plt.ylabel('Different ages ')
plt.title('Boxplot for age')

sns.catplot(x="year", kind="box", palette="ch:.25", data=df2,orient='v');
plt.ylabel('Year in which patient occures ')
plt.title('Boxplot for year')
sns.catplot(x="nodes", kind="box", palette="ch:.25", data=df2,orient='v');
plt.ylabel('No.of years treatment ')
plt.title('Boxplot for nodes')
sns.catplot(x="age", kind="count", palette="ch:.25", data=df2);
plt.title('Histogram for age column to count no. of patient with respect to age')

sns.catplot(x="year", kind="count", palette="ch:.25", data=df2);
plt.title('Histogram for year column to count no. of patient with respect to every year')

sns.catplot(x="nodes", kind="count", palette="ch:.25", data=df2);
plt.title('Histogram for nodes column to count no. of patient  are survive with respect to year of treatment')

sns.boxplot(x='status',y='age', data=df2)
plt.title('age vs status')

plt.show()
sns.boxplot(x='status',y='year', data=df2)
plt.title('year vs status')

plt.show()
sns.boxplot(x='status',y='nodes', data=df2)
plt.title('nodes vs status')

plt.show()
sns.violinplot( x='status', y='age', data=df2)
plt.title('age vs status')
plt.show() 
sns.violinplot( x='status', y='year', data=df2)
plt.title('year vs status')
plt.show() 
sns.violinplot( x='status', y='nodes', data=df2)
plt.title('no. of year treatment vs status')
plt.show() 

plt.close();
sns.set_style("whitegrid");
sns.pairplot(df2, hue="status", size=3);
plt.show()
print(df2.describe())
