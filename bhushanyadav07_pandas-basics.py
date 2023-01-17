import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

haberman = pd.read_csv("../input/haberman12/haberman12.csv")
haberman
print(haberman.shape)#trying to do from start
haberman['status'].value_counts() 
# first i read the all comment below the assignment
#replacing the status with yes or no
#it really takes too much time 
#haberman.status.map(dict(1:yes, 2: no))     
#haberman['status'] = haberman['status'].map({1:"yes", 2:"no"})
haberman['status'] = haberman['status'].map({1:"yes", 2:"no"})
haberman.head(20)
haberman.columns
haberman['status'].unique()
haberman.info() #checking updated info
#univariate analysis
sns.FacetGrid(haberman, hue="status", size=5) \
   .map(sns.distplot, "age") \
   .add_legend();
plt.show();
sns.FacetGrid(haberman, hue = "status" , size=5) \
    .map(sns.distplot, "year") \
    .add_legend();
plt.show();
#PDF, CDF, BOXPLOT, VOILIN PLOT
#PDF
import numpy as np
haberman_yes = haberman.loc[haberman["status"] == "yes"];
haberman_no = haberman.loc[haberman["status"] == "no"];


plt.figure(figsize=(20,5))
plt.subplot(141)#1=no.of row , 4=no.of columns 1=fig. number
counts,bin_edges=np.histogram(haberman_yes["age"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('AGE')
plt.legend(['PDF-age', 'CDF-age'])
plt.title('PDF-CDF of AGE Status = YES')

plt.subplot(142)#row 1 fig no 2
counts,bin_edges=np.histogram(haberman_yes["year"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('year')
plt.legend(['PDF-year', 'CDF-year'])
plt.title('PDF-CDF of year Status = YES')


plt.subplot(143)#row 1 fig 3
counts,bin_edges=np.histogram(haberman_no["age"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('AGE')
plt.legend(['PDF-age', 'CDF-age'])
plt.title('PDF-CDF of AGE Status = NO')


plt.subplot(144)#row 1 fig 4
counts,bin_edges=np.histogram(haberman_no["year"],bins=10,density=True)
pdf = counts/(sum(counts)) #formulae
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.ylabel("COUNT")
plt.xlabel('YEAR')
plt.legend(['PDF-year', 'CDF-year'])
plt.title('PDF-CDF of YEAR Status = NO')
plt.show();
plt.figure(figsize=(20,5))
plt.subplot(131)
sns.boxplot(x='status', y='nodes', data=haberman)
plt.title("box plot of NODES(y-axis) and STATUS(x-axis)")

plt.subplot(132)
sns.boxplot(x='status', y='age', data=haberman)
plt.title("box plot of AGE(y-axis) and STATUS(x-axis)")

plt.subplot(133)
sns.boxplot(x='status',y='year',data=haberman)
plt.title("box plot of YEAR(y-axis) and STATUS(x-axis)")
plt.show()
plt.figure(figsize=(20,5))
plt.subplot(131)
sns.violinplot(x='status', y='nodes', data=haberman ,size=8)
plt.title("violinplot of NODES(y-axis) and STATUS(x-axis)")

plt.subplot(132)
sns.violinplot(x='status', y='age', data=haberman, size=8)
plt.title("violinplot of AGE(y-axis) and STATUS(x-axis)")

plt.subplot(133)
sns.violinplot(x='status',y='year',data=haberman, size=8)
plt.title("violinplot of YEAR(y-axis) and STATUS(x-axis)")
plt.show()
#BI-variate analysis
#scatter plot
haberman.plot(kind='scatter', x='age', y='nodes') ;
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="status", size=5) \
   .map(plt.scatter, "age", "nodes") \
   .add_legend();
plt.show();
#pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="status", size=4);
plt.show()