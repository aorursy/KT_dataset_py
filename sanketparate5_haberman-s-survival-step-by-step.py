import numpy as np        
import pandas as pd       
import matplotlib.pyplot as plt         
import seaborn as sns                   
%matplotlib inline                      
import os                               
# Loading the Haberman's Survival Data Set
data= pd.read_csv("../input/habermans-survival-data-set/haberman.csv", names=["age", "Patient's Year of Operation","number of axillary nodes", "Survival Status"])
# How many data points and features in the dataset
data.shape
# Column names in the dataset
data.columns
data.head(5)
data['Survival Status'].unique()
# Number of Class in dataset
data['Survival Status'].value_counts()
# Calculate the Percentage of each Class
data["Survival Status"].value_counts()*100/data.shape[0]              #data.shape[0] is for total number of rows i.e.306
data.info()
Patient_Survived = data.loc[data["Survival Status"] ==1]
Patient_died = data.loc[data["Survival Status"] ==2]
data.tail(5)
data.describe()
sns.FacetGrid(data, hue="Survival Status", size=5).map(sns.distplot, "number of axillary nodes").add_legend()
plt.show()
import warnings                              # warnings
warnings.filterwarnings("ignore")
sns.FacetGrid(data, hue="Survival Status", size=5).map(sns.distplot, "age").add_legend()
plt.show()
sns.FacetGrid(data, hue="Survival Status", size=5).map(sns.distplot, "Patient's Year of Operation").add_legend()
plt.show()
Patient_Survived = data.loc[data["Survival Status"] ==1]
Patient_died = data.loc[data["Survival Status"] ==2]
data.tail()
counts, bin_edges = np.histogram(Patient_Survived["number of axillary nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)

# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['PDF for the patients who survive more than 5 years', 'CDF for the patients who survive more than 5 years'])
plt.show()
counts, bin_edges = np.histogram(Patient_died["number of axillary nodes"], bins=10, density = True)
pdf = counts/(sum(counts))
print(pdf)
print(bin_edges)

# Compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['PDF for the patients who died within 5 years', 'CDF for the patients who died within 5 years'])
plt.show()
print("Summary Statistics of Patients who survive more than 5 years or longer:")
Patient_Survived.describe()
print("Summary Statistics of Patients who died within 5 years:")
Patient_died.describe()
print("\nMedians:")
print(np.median(Patient_Survived["number of axillary nodes"]))

# Median with an Outlier
print(np.median(np.append(Patient_Survived["number of axillary nodes"],50)))
print(np.median(Patient_died["number of axillary nodes"]))

print("\n 90th Percentiles:")
print(np.percentile(Patient_Survived["number of axillary nodes"],90))
print(np.percentile(Patient_died["number of axillary nodes"], 90))



from statsmodels import robust
print("\n Median Absolute Deviation")
print(robust.mad(Patient_Survived["number of axillary nodes"]))
print(robust.mad(Patient_died["number of axillary nodes"]))

print("\nQuantiles:")
print(np.percentile(Patient_Survived["number of axillary nodes"], np.arange(0,100,25)))
print(np.percentile(Patient_died["number of axillary nodes"], np.arange(0,100,25)))
sns.boxplot(x="Survival Status", y ="age", data= data)
sns.boxplot(x="Survival Status", y ="Patient's Year of Operation", data= data)
sns.boxplot(x="Survival Status", y="number of axillary nodes", data=data)
plt.show()
sns.violinplot(x="Survival Status", y="age", data=data, size=7)
plt.show()
sns.violinplot(x="Survival Status", y="number of axillary nodes", data=data, size=7)
plt.show()
sns.violinplot(x="Survival Status", y="Patient's Year of Operation", data=data, size=7)
plt.show()
# 2D Density plot, contors-plot
sns.jointplot(x="age", y="Patient's Year of Operation", data=data, kind="kde")
plt.show()
import seaborn as sns
sns.scatterplot(x="age", y="number of axillary nodes", data=data)
# 2-D Scatter Plot with color-coding for each survival Status
# Here 'sns' corresponds to seaborn
sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Survival Status" ,size=6)\
.map(plt.scatter, "age", "number of axillary nodes")
plt.legend()
plt.show()
plt.close()
sns.set_style("whitegrid")
sns.pairplot(data, hue="Survival Status", size=3)
plt.show()
# The diagonal 