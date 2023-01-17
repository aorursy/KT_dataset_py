# Importing the libraries.
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
# Loading data
hm=pd.read_csv("../input/haberman.csv")

# Identifying shape : Number of data points(rows), features(columns)
display(hm.shape)
# Identify the features name
print(hm.columns)

# Renaming the columns as provided infomation
hm = hm.rename(columns = {'30':'age', '64':'operation_year','1':'positive_axillary_nodes','1.1':'survival_status'})
# Displaying first 10 indexex
display(hm.head(10))
# Getting the number of datapoints(class label) in each class. Class is explicitly informed.
# Here the class is survival_status and class labels are: '1', '2'
display(hm['survival_status'].value_counts())
# Getting some more information about data
hm.info()
# Getting some statistics about the dataset
# It is very useful some times
display(hm.describe())
# Plotting PDF for age of patients
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'age').add_legend()
plt.title("Age of patients and distribution")
plt.xlabel("Age")
plt.show()
# Plotting PDF for operation of year
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'operation_year').add_legend()
plt.title("Operation year and distribution")
plt.xlabel("Operation year (19xx)")
plt.show()
# Histogram of positive axillry nodes
sb.FacetGrid(hm, hue='survival_status', size=4).map(sb.distplot, 'positive_axillary_nodes').add_legend()
plt.title("Positive axillary nodes and distribution")
plt.xlabel("Positive axillay nodes")
plt.show()
# Plotting CDF of age of patients
# Here, bins indicates the number of bins data will be divided in.
counts, bin_edges = np.histogram(hm['age'], bins=10)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)

plt.title("CDF plot for age")
plt.xlabel("Age")
plt.ylabel("Probabilities")
plt.show()
# Plotting CDF of age of patients
counts, bin_edges = np.histogram(hm['positive_axillary_nodes'], bins=10)
pdf = counts/(sum(counts))
cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.title("CDF plot for positive axillary nodes")
plt.xlabel("Positive axillary nodes")
plt.ylabel("Probabilities")
plt.show()
# Univariate scatter plot
survived = hm.loc[hm["survival_status"] == 1]
non_survived = hm.loc[hm["survival_status"] ==2]

plt.plot(survived["age"], np.zeros_like(survived['age']), 'o', label="1")
plt.plot(non_survived["age"], np.zeros_like(non_survived["age"]), 'o', label="2")
plt.title("Age of survival and non survival")
plt.xlabel("Age")
plt.legend()

plt.show()
# Plotting the box-cox for survival status of age
sb.boxplot(x='survival_status', y='age', data=hm)
plt.title("Box-plot of age")
plt.xlabel("Survival status")
plt.ylabel("Age")
plt.show()
# Plotting box-cox for year of operation
sb.boxplot(x="survival_status", y="operation_year", data=hm)
plt.title("Box-plot of operation of year (19xx)")
plt.xlabel("Survival status")
plt.ylabel("Operation year")
plt.show()
# Plotting box-cox for positive axillary nodes
sb.boxplot(x="survival_status", y="positive_axillary_nodes", data=hm)
plt.title("Box-plot of positive axillary nodes")
plt.xlabel("Survival status")
plt.ylabel("Positive axillary nodes")
plt.show()
# Violin plots for age
sb.violinplot(x="survival_status", y="age", data=hm)
plt.title("Violin plots of age")
plt.xlabel("Survival status")
plt.ylabel("Age")
plt.show()
# Scatter plot of age and operations year
hm.plot(kind="scatter", x="age", y="operation_year")
plt.title("Scatter plot of and and year of operation")
plt.xlabel("Age")
plt.ylabel("Operation year")
plt.show()
# The same plot with different c
sb.set_style("whitegrid")
sb.FacetGrid(hm, hue="survival_status", size=5).map(plt.scatter, "age", "operation_year").add_legend()
plt.title("Age and Operation year")
plt.xlabel("Age")
plt.ylabel("Operation year")
plt.show()
# Scatter plot for Age and Positive axillary nodes
sb.set_style("whitegrid")
sb.FacetGrid(hm, hue="survival_status", size=5).map(plt.scatter, "age", "positive_axillary_nodes").add_legend()
plt.title("Age and Positive axillary nodes")
plt.xlabel("Age")
plt.ylabel("Positive axillary nodes")
plt.show()
sb.set_style("whitegrid")
sb.pairplot(hm, hue="survival_status", vars=["age", "operation_year", "positive_axillary_nodes"], size=4).fig.suptitle("Pair plot of age, operation year & positive axillary nodes")
plt.show()