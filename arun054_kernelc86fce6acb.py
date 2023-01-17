import warnings 

warnings.filterwarnings("ignore")
#import pandas, numpy, seaborn libraries

#import pyplot from matplotlib labrary

#import robust from statsmodels library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels import robust
#read the csv file into a dataframe

haberman = pd.read_csv("../input/haberman_1.csv")
#check how the dataset looks like

print(haberman.head())
#get the column names of the dataset

haberman.columns
#rename the column names accordingly

haberman = haberman.rename(columns={'30':'age','64':'yr_optd','1':'axil_nodes','1.1':'Survival'})
#get the number of points and number of features present in the dataset

print(haberman.shape)
#Get the number of labels and number of points per label

haberman["Survival"].value_counts()
#PDF of feature age

sns.FacetGrid(haberman,hue="Survival",height=6).map(sns.distplot,"age").add_legend()

plt.title('PDF of feature age')

plt.xlabel('age')

plt.ylabel('distribution')

plt.show()
sns.FacetGrid(haberman,hue="Survival",height=6).map(sns.distplot,'yr_optd').add_legend()

plt.title('PDF - year of operation (year - 1900)')

plt.ylabel("distribution")

plt.xlabel("year of operation (year - 1900)")

plt.show()
#PDF : number of axillary nodes 

sns.FacetGrid(haberman,hue="Survival",height=6).map(sns.distplot,"axil_nodes").add_legend()

plt.title('PDF - number of axillary nodes')

plt.ylabel("distribution")

plt.xlabel("number of axillary nodes")

plt.show()
#get the two labels into two seperate variables

survived_more_than_5_yrs = haberman.loc[haberman["Survival"]== 1]

survived_less_than_5_yrs = haberman.loc[haberman["Survival"]==2]
#CDF - year of operation



counts, bin_edges = np.histogram(survived_more_than_5_yrs['yr_optd'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=1")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=1")







counts, bin_edges = np.histogram(survived_less_than_5_yrs['yr_optd'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=2")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=2")

plt.legend()



plt.title('CDF - year of operation (year -1900)')

plt.ylabel("distribution")

plt.xlabel("year of operation (year - 1900)")



plt.show();
#CDF - age

counts, bin_edges = np.histogram(survived_more_than_5_yrs['age'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=1")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=1")







counts, bin_edges = np.histogram(survived_less_than_5_yrs['age'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=2")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=2")





plt.title('CDF : age')

plt.ylabel("distribution")

plt.xlabel("CDF : age")

plt.legend(loc='upper left')



plt.show();
#CDF - number of Axillary nodes

counts, bin_edges = np.histogram(survived_more_than_5_yrs['axil_nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=1")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=1")







counts, bin_edges = np.histogram(survived_less_than_5_yrs['axil_nodes'], bins=10, 

                                 density = True)

pdf = counts/(sum(counts))

print(pdf);

print(bin_edges)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf,label="pdf for Survival=2")

plt.plot(bin_edges[1:], cdf,label="cdf for Survival=2")

plt.legend()



plt.title('CDF : number of axillary nodes')

plt.ylabel("distribution")

plt.xlabel("number of axillary nodes")



plt.show();
#get the median and Median Absolute Deviation values

print("Median values for those who survived more than 5 yrs:")

print("Median Age: ", np.median(survived_more_than_5_yrs["age"]))

print("Median number of axillary nodes: ",np.median(survived_more_than_5_yrs["axil_nodes"]))

print("Median year of operation: ", np.median(survived_more_than_5_yrs["yr_optd"]))



print("Median values for those who survived less than 5 yrs:")

print("Median Age: ", np.median(survived_less_than_5_yrs["age"]))

print("Median number of axillary nodes: ",np.median(survived_less_than_5_yrs["axil_nodes"]))

print("Median year of operation: ",np.median(survived_less_than_5_yrs["yr_optd"]))
from statsmodels import robust

print("Median Absolute Deviation for those who survived more than 5 yrs:")

print("MAD age: ",robust.mad(survived_more_than_5_yrs["age"]))

print("MAD number of axillary nodes: ",robust.mad(survived_more_than_5_yrs["axil_nodes"]))

print("MAD year of operation: ",robust.mad(survived_more_than_5_yrs["yr_optd"]))



print("Median Absolute Deviation for those who survived less than 5 yrs:")

print("MAD age: ",robust.mad(survived_less_than_5_yrs["age"]))

print("MAD number of axillary nodes: ",robust.mad(survived_less_than_5_yrs["axil_nodes"]))

print("MAD year of operation: ",robust.mad(survived_less_than_5_yrs["yr_optd"]))
#Boxplot - age

sns.boxplot(x='Survival',y='age',data=haberman)

plt.title('Boxplot : age')

plt.show()
#Boxplot - year of operation

sns.boxplot(x="Survival",y='yr_optd',data=haberman)

plt.title('Boxplot : year of operation (year-1900)')

plt.show()
#Boxplot - axil nodes

sns.boxplot(x="Survival",y="axil_nodes",data=haberman)

plt.title('Boxplot : number of axillary nodes')

plt.show()
#Violinplot - age

sns.violinplot(x="Survival",y="age",data=haberman)

plt.title("Violinplot : age")

plt.show()
#Violinplot - axil nodes

sns.violinplot(x="Survival",y="axil_nodes",data=haberman)

plt.title('Violinplot : number of axillary nodes')

plt.show()
#Violinplot - year of operation

sns.violinplot(x="Survival",y="yr_optd",data=haberman)

plt.title("Violinplot : year of operation(year - 1900)")

plt.show()
#Scatter plot - age, year of operation

sns.set_style('whitegrid');

sns.FacetGrid(haberman,hue='Survival',height=6).map(plt.scatter,'age','yr_optd').add_legend()

plt.title("Scatter plot : age vs year of operation (year-1900)")

plt.show()
#Scatter plot - number of axillary nodes, age

sns.set_style('whitegrid');

sns.FacetGrid(haberman,hue='Survival',height=6).map(plt.scatter,'age','axil_nodes').add_legend()

plt.title("Scatter plot : age vs number of axillary nodes")

plt.show()
#Scatter plot - year of operation, number of axillary nodes

sns.set_style('whitegrid');

sns.FacetGrid(haberman,hue='Survival',height=6).map(plt.scatter,'axil_nodes','yr_optd').add_legend()

plt.title("Scatter plot : number of axillary nodes vs year of operation(year-1900)")

plt.show()
#pair plot

plt.close()



sns.set_style('whitegrid');

sns.pairplot(haberman,hue="Survival",vars=['age','yr_optd','axil_nodes'],height = 3);

plt.suptitle("Pairplot : Age vs year of operation vs number of axillary nodes",y=1.02)

plt.show()
#Joint plot - age, year of operation

sns.jointplot(x="age",y="yr_optd",data=haberman,kind="kde")

plt.suptitle("Jointplot : age vs year of operation(year-1900)",y=1.02)

plt.show()
#Joint plot - age, axillary nodes

sns.jointplot(x="age",y="axil_nodes",data=haberman,kind="kde")

plt.suptitle("Jointplot : age vs number of axillary nodes",y=1.02)

plt.show()
#Joint plot - number of axillary nodes, year of operation

sns.jointplot(x="axil_nodes",y="yr_optd",data=haberman,kind="kde")

plt.suptitle("Jointplot : number of axillary nodes vs year of operation(year-1900)",y=1.02)

plt.show()