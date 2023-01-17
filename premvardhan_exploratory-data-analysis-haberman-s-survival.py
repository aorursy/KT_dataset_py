import pandas as pd # Data analysis and manipulation 
import numpy as np # Numerical operations

# Data visualization
import matplotlib.pyplot as plt 
import seaborn as sns
# To load the data into DataFrame(It is 2d mutable, heterogeneous tabular data structure with labeled axes)

#df = pd.read_csv("haberman.csv")
df = pd.read_csv('../input/haberman.csv')
df.head()
# To know number of features/columns.

df.columns
# Number of rows(data-points) and columns(features).

df.shape
# To rename the column name for better understanding
# Both of the ways we can rename the columns name

#df.columns = ["age", "operation_year", "axillary_lymph_node", "survival_status"]

#or

df = df.rename(columns = {"30" : "age", "64" : "operation_year", "1" : "axillary_lymph_node", "1.1" : "survival_status"})

# It gives you the top 5 rows(data-points).

df.head()
# To look at the last 5 rows(data-points), we can also specify that how many data-points we want to see.

df.tail(7)
# To know about data summary

df.info()
# To know statistical summary of data which is very important

df.describe()
# To know number of data-points for each class.
# As it is not balanced dataset, it is imbalanced dataset because the number of data-points for both of the class are significantly different.
# we will see how to handle imbalanced data later

df.survival_status.value_counts()
# Here, we are using age feature to generate pdf()
# pdf(smoothed form of histogram)
# pdf basically shows, how many of points lies in some interval

sns.FacetGrid(df, hue = "survival_status", size = 5).map(sns.distplot, "age").add_legend()
plt.title("Histogram of age")
plt.ylabel("Density")
plt.show()
sns.FacetGrid(df, hue = "survival_status", size = 5). map(sns.distplot, "operation_year").add_legend()
plt.title("Histogram of operation_year")
plt.ylabel("Density")
plt.show()
sns.FacetGrid(df, hue = "survival_status", size = 5).map(sns.distplot, "axillary_lymph_node").add_legend()
plt.title("Histogram of axillary_lymph_node")
plt.ylabel("Density")
plt.show()
# one = df.loc[df["survival_status"] == 1]
# two = df.loc[df["survival_status"] == 2]
# cdf gives you cummulative probability associated with a function
# Cumulative sum of area under curve upto gives you cdf
# Here, Class 1 means survived
# Class 2 means not survived
one = df.loc[df["survival_status"] == 1]
two = df.loc[df["survival_status"] == 2]
label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for age")
plt.xlabel("age")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(two["age"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show()


# 53-58 
label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["operation_year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(two["operation_year"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for operation_year")
plt.xlabel("operation_year")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(label)

plt.show();

label = ["pdf of class 1", "cdf of class 1", "pdf of class 2", "cdf of class 2"]
counts, bin_edges = np.histogram(one["axillary_lymph_node"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

counts, bin_edges = np.histogram(two["axillary_lymph_node"], bins=10, density = True)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)
plt.title("pdf and cdf for axillary_lymph_node")
plt.xlabel("axillary_lymph_node")
plt.ylabel("% of person's")
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.show();

# boxplot gives you the statistical summery of data
# Rectangle represent the 2nd and 3rd quartile (horizontal line either side of the rectangle)
# The horizontal line inside box represents median
# We can add title in box plot using either way
# plt.title("Box plot for survival_status and age") or set_title("")

sns.boxplot(x = "survival_status", y = "age", hue = "survival_status", data = df).set_title("Box plot for survival_status and age")
plt.show()
sns.boxplot(x = "survival_status", y = "operation_year", hue = "survival_status", data = df).set_title("Box plot for survival_status and operation_year")
plt.show()
sns.boxplot(x = "survival_status", y = "axillary_lymph_node", hue = "survival_status", data = df).set_title("Box plot for survival_status and axillary_lymph_node")
plt.show()
# The violin plot shows the full distribution of the data.
# It is combination of box plot and histogram
# central dot represents median

sns.violinplot(x = "survival_status", y = "age", hue = "survival_status", data = df)
plt.title("Violin plot for survival_status and age")
plt.show()
sns.violinplot(x = "survival_status", y = "operation_year", hue = "survival_status", data = df)
plt.title("Violin plot for survival_status and operation_year")
plt.show()
sns.violinplot(x = "survival_status", y = "axillary_lymph_node", hue = "survival_status", data = df)
plt.title("Violin plot for survival_status and axillary_lymph_node")
plt.show()
# 1-d scatter plot

one = df.loc[df["survival_status"] == 1]
two = df.loc[df["survival_status"] == 2]
plt.plot(one["age"], np.zeros_like(one["age"]), 'o', label = "survival_status\n" "1")
plt.plot(two["age"], np.zeros_like(two["age"]), 'o', label = "2")
plt.title("1-D scatter plot for age")
plt.xlabel("age")
plt.legend()
plt.show()
# 2-d scatter plot

df.plot(kind = "scatter", x = "age", y = "operation_year")
plt.title("2-D scatter plot of age")
plt.show()
# 2d scatter plot with color coding for each class

sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "survival_status", size = 4).map(plt.scatter, "age", "operation_year").add_legend()
plt.title("2-D scatter plot for age and operation_year")
plt.show()

# 2d scatter plot 

sns.set_style("whitegrid")
sns.FacetGrid(df, hue = "survival_status", size = 4).map(plt.scatter, "age", "axillary_lymph_node").add_legend()
plt.title("2-D scatter plot for age and axillary_lymph_node")
plt.show()
# To convert data from numerical type to object type

# df["survival_status"] = df["survival_status"].astype(str)
# df["survival_status"].dtype
# To change data from numerical into object type
# changeing the numerical variable 1 into string "survived" and 2 into "not_survived"
# Altough, we can directly pass desired no of variable to plot using vars(list of variable name) parameter. which is used below of this cell.

#df["survival_status"] = df["survival_status"].apply(lambda x : "survived" if x == 1 else "not_survived")
#df["survival_status"].dtype
#Here, we are generating pairplot based on survival_status
#We use pair plot where the dimenstionality of data is less.
#In our case we have only 4 dimension data. So, we can use pairplot.

sns.set_style("whitegrid")
sns.pairplot(df, hue = "survival_status", vars = ["age", "operation_year", "axillary_lymph_node"], size = 3)
plt.suptitle("pair plot of age, operation_year and axillary_node")
plt.show()