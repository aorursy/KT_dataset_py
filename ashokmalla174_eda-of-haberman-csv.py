import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
df=pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv",names=["age","year","nodes","status"])
#top 5 rows
#last 5 rows
df.head
df.tail
#no of columns and rows in the data set
df.shape
#column names
df.columns
#Null values in the data set
df.isnull().sum()
df['status'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
# unique values with type in data set 
df["age"].unique()
df.describe()
#types of variable in the data
df.info()
#scatter plot with pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, vars = ["age", "year", "nodes"], hue="status",height=3);
plt.show()
sns.boxplot(x='status',y='age', data=df)
plt.title("Box Plot of status of patients survived or not ", fontsize=20)
plt.show()
sns.boxplot(x='status',y='nodes', data=df)
plt.title("Box plot of nodes of patient survived or not",fontsize=20)
plt.show()
#pdf abd cdf refer by(https://www.kaggle.com/gokulkarthik/haberman-s-survival-exploratory-data-analysis)
plt.figure(figsize=(15,3))
for idx, feature in enumerate(list(df.columns)[:-1]):
    plt.subplot(1, 3, idx+1)
    print("********* "+feature+" *********")
    counts, bin_edges = np.histogram(df[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)
    plt.legend(('PDF','CDF'))
#vailin plot is like a box plot
sns.violinplot(x="status",y="age",data=df)
plt.title("voilin plot of age vs status",fontsize=25)
sns.violinplot(x="status",y="year",data=df)
plt.title("voilin plot of year vs status",fontsize=25)
sns.violinplot(x="status",y="nodes",data=df)
plt.title("voilin plot of Nodes vs status",fontsize=25)

sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height=7 ).map(sns.distplot , "age").add_legend();
plt.title("Histogram of age of patients",fontsize=20)
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height=7 ).map(sns.distplot , "year").add_legend();
plt.title("Histogram of year of patients ",fontsize=20)
plt.show()
sns.set_style("whitegrid");
sns.FacetGrid(df , hue = "status" , height =7 ).map(sns.distplot , "nodes").add_legend();
plt.title("Histogram of status of patients",fontsize=20)
plt.show()

counts, bin_edges = np.histogram(df['age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


# virginica
counts, bin_edges = np.histogram(df['year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)


#versicolor
counts, bin_edges = np.histogram(df['nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(('PDF','CDF'))

plt.show();