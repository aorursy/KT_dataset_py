import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from matplotlib import pyplot as plt
from statsmodels import robust
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pylab
#Importing the dataset
df_cancer = pd.read_csv("../input/haberman.csv")
#Looking at the shape of the data
df_cancer.shape

#This dataset has 306 rows and 4 columns
#Looking at first few rows of the dataset
df_cancer.head()
df_cancer.describe()
counts_1,bins_1=np.histogram(df_cancer["Age"][df_cancer.survival_status==1],bins=10,density=True)
print(counts_1)
print(bins_1)
pdf_1=counts_1*np.diff(bins_1)
cdf_1=np.cumsum(pdf_1)
plt.plot(bins_1[1:],cdf_1,label="cdf_survived")

counts_2,bins_2=np.histogram(df_cancer["Age"][df_cancer.survival_status==2],bins=10,density=True)
print(counts_2)
print(bins_2)
pdf_2=counts_2*np.diff(bins_2)
cdf_2=np.cumsum(pdf_2)
plt.plot(bins_2[1:],cdf_2,label="cdf_not_survived")
plt.xlabel("Age")
plt.legend()
plt.title("CDF for Age")
plt.show()
#pdf for Age
sns.distplot(df_cancer.Age)
plt.title("PDF for Age")
plt.show()
#Probability density plot for Age by survival class.
sns.set_style("whitegrid")
sns.FacetGrid(df_cancer,hue="survival_status",size=5)\
   .map(sns.distplot,"Age")\
   .add_legend();
plt.title("PDF for Age by Survival class")
plt.show()
#cdf curve for "positive_axillary_nodes".
counts,bins=np.histogram(df_cancer["positive_axillary_nodes"],bins=10,density=True)
print(counts)
print(bins)
pdf=counts*np.diff(bins)
cdf=np.cumsum(pdf)
plt.plot(bins[1:],cdf,label="cdf_survived")
plt.xlabel("positive_axillary_nodes")
plt.title("CDF-positive_axillary_nodes")
plt.show()
#cdf curve for "positive_axillary_nodes" by survival status.
counts_1,bins_1=np.histogram(df_cancer["positive_axillary_nodes"][df_cancer.survival_status==1],bins=10,density=True)
print(counts_1)
print(bins_1)
pdf_1=counts*np.diff(bins)
cdf_1=np.cumsum(pdf_1)
plt.plot(bins_1[1:],cdf_1,label="cdf_survived")

counts_2,bins_2=np.histogram(df_cancer["positive_axillary_nodes"][df_cancer.survival_status==2],bins=10,density=True)
print(counts_2)
print(bins_2)
pdf_2=counts_2*np.diff(bins_2)
cdf_2=np.cumsum(pdf_2)
plt.plot(bins_2[1:],cdf_2,label="cdf_not_survived")
plt.xlabel("positive_axillary_nodes")
plt.legend()
plt.title("CDF-positive_axillary_nodes")
plt.show()
#Probability Density plot for positive_axillary_nodes
sns.set_style("whitegrid")
sns.FacetGrid(df_cancer,hue="survival_status",size=8)\
   .map(sns.distplot,"positive_axillary_nodes")\
   .add_legend();
plt.show()
#cdf curve for "year" by survival status.
counts_1,bins_1=np.histogram(df_cancer["year"][df_cancer.survival_status==1],bins=10,density=True)
print(counts_1)
print(bins_1)
pdf_1=counts_1*np.diff(bins_1)
cdf_1=np.cumsum(pdf_1)
plt.plot(bins_1[1:],cdf_1,label="cdf_survived")

counts_2,bins_2=np.histogram(df_cancer["year"][df_cancer.survival_status==2],bins=10,density=True)
print(counts_2)
print(bins_2)
pdf_2=counts_2*np.diff(bins_2)
cdf_2=np.cumsum(pdf_2)
plt.plot(bins_2[1:],cdf_2,label="cdf_not_survived")
plt.xlabel("year")
plt.title("CDF-year of surgery")
plt.legend()
plt.show()
#Probability Density plot for surgery_year
sns.set_style("whitegrid")
sns.FacetGrid(df_cancer,hue="survival_status",size=8)\
   .map(sns.distplot,"year")\
   .add_legend();
plt.show()
#Generating the pairplot
sns.set_style("whitegrid")
sns.pairplot(df_cancer,vars=["Age","year","positive_axillary_nodes"],hue="survival_status",size=4)
plt.show()
#Scatterplot positive_axillary_nodes VS Age
sns.set_style("whitegrid");
sns.FacetGrid(df_cancer,hue="survival_status",size=5)\
   .map(plt.scatter,"Age","positive_axillary_nodes")\
   .add_legend();
plt.title("positive_axillary_nodes VS Age")
plt.show()
#Contour plot for Age and positive_axillary_nodes
sns.jointplot(x="Age",y="positive_axillary_nodes",data=df_cancer,kind="kde",size=6)
plt.show()
#Now generating the Age VS positive_axillary_node scatterplot only for age>50
#to see the effect of old age on survival
sns.set_style("whitegrid");
sns.FacetGrid(df_cancer[df_cancer.Age>50],hue="survival_status",size=5)\
   .map(plt.scatter,"Age","positive_axillary_nodes")\
   .add_legend();
plt.show()
df_50 = df_cancer[df_cancer.Age>50]
df_cancer[df_cancer.Age>50]
counts_1,bins_1=np.histogram(df_50["Age"][df_50.survival_status==1],bins=10,density=True)
print(counts_1)
print(bins_1)
pdf_1=counts_1*np.diff(bins_1)
cdf_1=np.cumsum(pdf_1)
plt.plot(bins_1[1:],cdf_1,label="cdf_survived")

counts_2,bins_2=np.histogram(df_50["Age"][df_50.survival_status==2],bins=10,density=True)
print(counts_2)
print(bins_2)
pdf_2=counts_2*np.diff(bins_2)
cdf_2=np.cumsum(pdf_2)
plt.plot(bins_2[1:],cdf_2,label="cdf_not_survived")
plt.xlabel("Age")
plt.legend()
plt.title("CDF for (Age>50)")
plt.show()
counts_1,bins_1=np.histogram(df_50["positive_axillary_nodes"][df_50.survival_status==1],bins=10,density=True)
print(counts_1)
print(bins_1)
pdf_1=counts_1*np.diff(bins_1)
cdf_1=np.cumsum(pdf_1)
plt.plot(bins_1[1:],cdf_1,label="cdf_survived")

counts_2,bins_2=np.histogram(df_50["positive_axillary_nodes"][df_50.survival_status==2],bins=10,density=True)
print(counts_2)
print(bins_2)
pdf_2=counts_2*np.diff(bins_2)
cdf_2=np.cumsum(pdf_2)
plt.plot(bins_2[1:],cdf_2,label="cdf_not_survived")
plt.xlabel("positive_axillary_nodes")
plt.legend()
plt.title("CDF for positive nodes (Age>50)")
plt.show()
#positive_axillary_nodes VS year scatterplot
sns.set_style("whitegrid");
sns.FacetGrid(df_cancer,hue="survival_status",size=5)\
   .map(plt.scatter,"year","positive_axillary_nodes")\
   .add_legend();
plt.title("positive_axillary_nodes VS surgery_year")
plt.show()
sns.violinplot(x="survival_status",y="positive_axillary_nodes",data=df_cancer)
plt.show()