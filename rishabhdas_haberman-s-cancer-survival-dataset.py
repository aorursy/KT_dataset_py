import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

habers=pd.read_csv('../input/haberman.csv')
#(Q) What are the column names?
#(Q) How many data points and features?
habers.info()
habers.shape
habers['status'].value_counts()
sns.set_style('whitegrid')
sns.pairplot(habers,hue='status',size=3)
plt.show()
habers["status"] = habers["status"].apply(lambda y: "Survived" if y == 1 else "Dead")
survived=habers.loc[habers['status'] == 'Survived']
dead=habers.loc[habers['status'] == 'Dead']

plt.plot(survived['detected_pos_nodes'], np.zeros_like(survived['detected_pos_nodes']),'o')
plt.plot(dead['detected_pos_nodes'], np.zeros_like(dead['detected_pos_nodes']),'x')
plt.show()
survived.describe()
dead.describe()
sns.FacetGrid(habers, hue='status', size=5) \
    .map(sns.distplot, 'age') \
    .add_legend()

plt.show()
sns.FacetGrid(habers, hue='status', size=5) \
    .map(sns.distplot, 'year_of_op') \
    .add_legend()

plt.show()
sns.FacetGrid(habers, hue='status', size=5) \
    .map(sns.distplot, 'detected_pos_nodes') \
    .add_legend()

plt.show()
#CDF of Auxilary nodes
#Survived

counts, bin_edges = np.histogram(survived['detected_pos_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF of Age
#Survived
counts, bin_edges = np.histogram(survived['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF of Year of Operation
#Survived

counts, bin_edges = np.histogram(survived['year_of_op'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF of Auxilary Nodes
#Dead

counts, bin_edges = np.histogram(dead['detected_pos_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF of Age
#Dead

counts, bin_edges = np.histogram(dead['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
#CDF of Year of Operation
#Dead

counts, bin_edges = np.histogram(dead['year_of_op'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



plt.show();
sns.boxplot(x='status',y='age', data=habers)
plt.show()
sns.boxplot(x='status',y='year_of_op', data=habers)
plt.show()
sns.boxplot(x='status',y='detected_pos_nodes', data=habers)
plt.show()
