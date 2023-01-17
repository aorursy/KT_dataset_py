import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



'''downlaod Habernets csv  from https://www.kaggle.com/gilsousa/habermans-survival-data-set/version/1#haberman.csv'''

#Load habernets into a pandas dataFrame.

data = pd.read_csv("../input/haberman.csv", names=['age', 'year', 'nodes', 'status'])#loads the data in haberman.csv file into data
#Number of data points(rows) and Features(columns)

print(data.shape)
#Column names in our dataset

print (data.columns)
'''to change the names of the columns

data.columns=['age','year','nodes','Status']

print(data.columns)

'''
#to get information about the data set like the type of the data in the file(int)

#we can also check weather their is a null value or not by looking at the type if there is null value if it is 'object' it has NULL value

data.info()
#to check the number of Null values

np.sum(data.isna())
#to get the details like number of observations, min,max,25%,50%,75% ,mean,std

data.describe()
#to check number of patients survived 5 years or longer and number of patients died within 5 year

data['status'].value_counts()

#slightly Balanced data Set
#pie chart representation

status=data['status'].value_counts()



labels='Patients survived 5 years or longer','patients died within 5 years'

sizes=[status[1],status[2]]

fig1, ax1 = plt.subplots()

ax1.pie(sizes,labels=labels)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
#to find the corelation between the columns

data.corr()
sns.set_style("whitegrid");

sns.FacetGrid(data,hue='status',size=6).map(plt.scatter,'age','nodes').add_legend()

plt.show()

#1->the patient survived 5 years or longer

#2->patient died within 5 year
# pairwise scatter plot: Pair-Plot.

plt.close();

sns.set_style("whitegrid");

sns.pairplot(data,hue='status',vars=['age','year','nodes'],size=5,diag_kind='kde');

plt.legend()

plt.show() 

# NOTE: the diagnol elements are PDFs for each feature. PDFs are expalined below.
counts,bin_edges=np.histogram(data['age'],bins=10,density=True)

pdf=counts/(sum(counts))

#print(pdf)

#print(bin_edges) 

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()
# Or we can use the percentile concept to get the same information

print(np.percentile(data['age'],np.arange(0,100,25)))#for calculating quantiles(0,25,50,75)

print(np.percentile(data['age'],95))#for caluculating 95th percentile
counts,bin_edges=np.histogram(data['nodes'],bins=10,density=True)

pdf=counts/(sum(counts))

#print(pdf)

#print(bin_edges) 

cdf=np.cumsum(pdf)

plt.plot(bin_edges[1:],pdf)

plt.plot(bin_edges[1:],cdf)

plt.show()



print(np.percentile(data['nodes'],95))#for caluculating 95th percentile
sns.FacetGrid(data,hue='status',size=5).map(sns.distplot,'age').add_legend()

plt.show()
sns.FacetGrid(data, hue='status', size=5).map(sns.distplot, 'nodes').add_legend();

plt.show();
sns.FacetGrid(data,hue='status',size=5).map(sns.distplot,'year').add_legend();

plt.show()
#NOTE: IN the plot below, a technique call inter-quartile range is used in plotting the whiskers. 

#Whiskers in the plot below donot correposnd to the min and max values.



#Box-plot can be visualized as a PDF on the side-ways.



sns.boxplot(x='status',y='age', data=data)

plt.show()
sns.boxplot(x='status',y='nodes', data=data)

plt.show()
sns.violinplot(x="status", y="age", data=data, size=10)

plt.show()
sns.violinplot(x="status", y="nodes", data=data, size=10)

plt.show()