#import the library numpy matplotlib,seaborn,pandas

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

%matplotlib inline
# Download the Haberman's Survival Data Set from "https://www.kaggle.com/gilsousa/habermans-survival-data-set" 

#load the Haberman's Survival Data Set
df = pd.read_csv("haberman.csv") # assume that i am loading the data in current dircetory 

# here columns are not more expressable so here i am setting the columns names

col = ['Age','Year-of-operation','Positive-auxilary-nodes','Status']
df = pd.read_csv("haberman.csv",names = col)
df.head()


# how many data points and features are there.
df.shape


# Now what are the columns name of our data set

print(df.columns)
# print the number of class in data set.
df['Status'].value_counts()

# Here  is 2 class and this data set is imbalnced because class1 have 225 data points and class2 have 81 data points


# count,mean ,standard deviation(std),minimum(min),First quartile(25%),Second quartile(50%),Third quartile(75%), maximum(max) for each
# feature of dataset.

df.describe()

# calculate the percentage  of each class
df['Status'].value_counts()*100/df.shape[0]
survived = df.loc[df['Status']== 1]
not_survived = df.loc[df['Status'] == 2]
print("Patients survied 5 years or more")
survived.describe()

print("Patients died within 5 years")
not_survived.describe()

survived_status = survived['Age'].describe()
not_survived_status =not_survived['Age'].describe()

# create dataframe to store the survived_status and not_survived_status statistics.
df1_age = pd.DataFrame(data={'Survived':survived_status,'Died':not_survived_status})


df1_age

sns.FacetGrid(df,hue = 'Status',size =6).map(sns.distplot,'Age',bins = 20).add_legend()
plt.title("Age Feature")
plt.show()

# draw the boxplot for age feature:
sns.boxplot(x = 'Status',y = 'Age',data = df)

#  drwa the violoin plot for age feature:
sns.violinplot(x = 'Status',y = 'Age',data = df,size = 6)

# now plot the pdf and cdf for Age

%matplotlib inline
# PDF & CDF
# compute pdf & cdf for survived

counts, bin_edges = np.histogram(survived['Age'],bins = 20,density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,"ro-",label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,"r*-",label = "not survived cdf")


# compute pdf & cdf for not_survived 

counts,bin_edgs = np.histogram(not_survived['Age'],bins= 20,density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,'bo-',label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,'b*-',label = "not survived cdf")
plt.legend()
plt.xlabel("Age")

plt.show()

survived_status = survived['Year-of-operation'].describe()
not_survived_status =not_survived['Year-of-operation'].describe()

# create dataframe to store the survived_status and not_survived_status statistics.
df1_year = pd.DataFrame(data={'Survived':survived_status,'Died':not_survived_status})


df1_year
sns.FacetGrid(df,hue = 'Status',size =6).map(sns.distplot,'Year-of-operation',bins = 10).add_legend()
plt.title("Year-of-operation Feature")
plt.show()
# draw the boxplot for year-of-operation feature:

sns.boxplot(x = 'Status',y = 'Year-of-operation',data = df)
# draw th violinplot for year-of-operation feature:

sns.violinplot(x = 'Status',y = 'Year-of-operation',data = df)
plt.show()
# now calculate pdf & cdf for year-of-operation 

%matplotlib inline
# PDF & CDF
# compute pdf & cdf for survived

counts, bin_edges = np.histogram(survived['Year-of-operation'],bins = 20,density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,"ro-",label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,"r*-",label = "not survived cdf")


# compute pdf & cdf for not_survived 

counts,bin_edgs = np.histogram(not_survived['Year-of-operation'],bins= 20,density = True)
pdf = counts/sum(counts)
print(pdf)
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,'bo-',label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,'b*-',label = "not survived cdf")
plt.legend()
plt.xlabel("Year-of-opeartion")

plt.show()

survived_status = survived['Positive-auxilary-nodes'].describe()
not_survived_status =not_survived['Positive-auxilary-nodes'].describe()

# create dataframe to store the survived_status and not_survived_status statistics.
df1_nodes = pd.DataFrame(data={'Survived':survived_status,'Died':not_survived_status})


df1_nodes
sns.FacetGrid(df,hue = 'Status',size =7).map(sns.distplot,'Positive-auxilary-nodes',bins = 10).add_legend()
plt.title("Positive-auxilary-nodes")
plt.show()
# draw the violinplot for Positive-auxilary-nodes

sns.violinplot(x= 'Status',y = 'Positive-auxilary-nodes',data = df)
plt.show()
# draw the boxplot for Positive-auxilary-nodes

sns.boxplot(x = 'Status',y = 'Positive-auxilary-nodes',data = df)
plt.show()

# now calculate the pdf & cdf for positive-auxilary nodes.

%matplotlib inline
# PDF & CDF
# compute pdf & cdf for survived

counts, bin_edges = np.histogram(survived['Positive-auxilary-nodes'],bins = 55,density = True)
pdf = counts/sum(counts)
# print(pdf)
# print(bin_edges)
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,"--g",label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,"g",label = "not survived cdf")


# compute pdf & cdf for not_survived 

counts,bin_edgs = np.histogram(not_survived['Positive-auxilary-nodes'],bins= 55,density = True)
pdf = counts/sum(counts)
#print(pdf)
# print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf,'--r',label = "Survived pdf")
plt.plot(bin_edges[1:],cdf,'r',label = "not survived cdf")
plt.legend()
plt.xlabel("Positive-auxilary-nodes")

plt.show()


# plot the 3*3 grid for haberman dataset

 
# sns.set_style("whitegrid")
sns.pairplot(df,hue = 'Status',vars = ['Age','Year-of-operation','Positive-auxilary-nodes'],diag_kind = 'kde',
           size =4);
plt.show()

