#Enivronment Configration 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #data visualization
import seaborn as sns
#loading the dataset into dataframe

haberman_df=pd.read_csv('../input/haberman.csv')
print(haberman_df.head())
#Column names are not readable from above view 

haberman_df.columns = ["age", "operation_year", "axillary_lymph_node", "survival_status"]
print(haberman_df.head())
print(haberman_df.info())
      
haberman_df.shape
haberman_df.describe()
for idx, feature in enumerate(haberman_df.columns[:-1]):
    g = sns.FacetGrid(haberman_df , hue="survival_status" , size = 5)
    g.map(sns.distplot , feature , label = feature).add_legend()
    plt.show()
plt.figure(figsize=(5,20))
for index, feature in enumerate(haberman_df.columns[:-1]):
    plt.subplot(3,1,index+1)
    counts, bin_edges = np.histogram(haberman_df[feature] , bins = 10 , density = True)
    pdf = counts/sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf)
    plt.plot(bin_edges[1:],cdf)
    plt.xlabel(feature)
fig, ax = plt.subplots(1,3, figsize = (15,5))
for idx, feature in enumerate(haberman_df.columns[:-1]):
    sns.boxplot(x = 'survival_status' , y = feature , data = haberman_df , ax = ax[idx])
plt.show()
fig, ax = plt.subplots(1, 3, figsize= (15,5))
for idx, feature in enumerate(haberman_df.columns[:-1]):
    sns.violinplot(x = 'survival_status' , y = feature , data = haberman_df , ax = ax[idx])
plt.show()
sns.set_style('whitegrid')
g = sns.FacetGrid(haberman_df , hue = 'survival_status' , size = 5 )
g.map(plt.scatter , "age", "operation_year")
g.add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()
sns.set_style('whitegrid')
g = sns.FacetGrid(haberman_df , hue = 'survival_status' , size = 5 )
g.map(plt.scatter , "age", "axillary_lymph_node")
g.add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()
sns.set_style('whitegrid')
sns.pairplot(haberman_df , hue = 'survival_status' , vars = ['age' , "operation_year", "axillary_lymph_node"] , size = 4)
plt.suptitle("Pair plot of age, opertion year and axillary lymph node with survival status")
plt.show()
