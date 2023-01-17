# import the required Python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

import os
print(os.listdir('../input'))

# Load Dataset into Pandas Dataframe after downloading from Kaggle. Dataset kept in the same folder as the Jupyter notebook.
Data_Cancer = pd.read_csv('../input/haberman.csv',header = None)

# Column names are not defined, so setting header = None
# Check the number of data points and features
Data_Cancer.shape

# Dataset contains 306 rows and 4 columns
# Setting column names as per Dataset Description
Data_Cancer.columns = ['Age', 'Treatment_Year', 'No_of_Positive_Lymph_Nodes', 'Survival_Status_After_5_Years']
# Checking he first few datapoints
Data_Cancer.head()


# Mapping the Survival Status value 1 to 'Survived' and 2 to 'Died' for better readability
Data_Cancer['Survival_Status_After_5_Years'] = Data_Cancer['Survival_Status_After_5_Years'].map({1:'Survived',2:'Died'})
Data_Cancer.head()
# Getting an high level idea about the Dataset
Data_Cancer.describe()
Data_Cancer['Survival_Status_After_5_Years'].value_counts()
# Plotting PDF and CDF to identify features useful for classification
Data_Cancer_Survived = Data_Cancer[Data_Cancer['Survival_Status_After_5_Years'] == 'Survived']
Data_Cancer_Died = Data_Cancer[Data_Cancer['Survival_Status_After_5_Years'] == 'Died']
# On Age

sbn.FacetGrid(Data_Cancer, hue="Survival_Status_After_5_Years", size=5) \
   .map(sbn.distplot, "Age") \
   .add_legend();
plt.show();
# on Treatment Year

sbn.FacetGrid(Data_Cancer, hue="Survival_Status_After_5_Years", size=5) \
   .map(sbn.distplot, "Treatment_Year") \
   .add_legend();
plt.show();
# on No of POsitive Lymph Nodes

sbn.FacetGrid(Data_Cancer, hue="Survival_Status_After_5_Years", size=5) \
   .map(sbn.distplot, "No_of_Positive_Lymph_Nodes") \
   .add_legend();
plt.show();
# Plots of CDF and PDF of Age for various Survival Status.

# Survived Status
counts, bin_edges = np.histogram(Data_Cancer_Survived['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



# Died Status
counts, bin_edges = np.histogram(Data_Cancer_Died['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Age')




plt.show()
# Plots of CDF and PDF of TReatment_Year for various Survival Status.

# Survived Status
counts, bin_edges = np.histogram(Data_Cancer_Survived['Treatment_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)




# Died Status
counts, bin_edges = np.histogram(Data_Cancer_Died['Treatment_Year'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('Treatment_Year')



plt.show()
# Plots of CDF and PDF of No_of_Positive_Lymph_Nodes for various Survival Status.

# Survived Status
counts, bin_edges = np.histogram(Data_Cancer_Survived['No_of_Positive_Lymph_Nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



# Died Status
counts, bin_edges = np.histogram(Data_Cancer_Died['No_of_Positive_Lymph_Nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel('No_of_Positive_Lymph_Nodes')



plt.show()
# Box Plot for Age

sbn.boxplot(x='Survival_Status_After_5_Years',y='Age', data=Data_Cancer)
plt.show()
# Box Plot for Treatment_Year

sbn.boxplot(x='Survival_Status_After_5_Years',y='Treatment_Year', data=Data_Cancer)
plt.show()
# Box Plot for No_of_Positive_Lymph_Nodes

sbn.boxplot(x='Survival_Status_After_5_Years',y='No_of_Positive_Lymph_Nodes', data=Data_Cancer)
plt.show()
# Violin Plot for Age
sbn.violinplot(x="Survival_Status_After_5_Years", y="Age", data=Data_Cancer, size=8)
plt.show()
# Violin Plot for Treatment_Year
sbn.violinplot(x="Survival_Status_After_5_Years", y="Treatment_Year", data=Data_Cancer, size=8)
plt.show()
# Violin Plot for No_of_Posiive_Lymph_Nodes
sbn.violinplot(x="Survival_Status_After_5_Years", y="No_of_Positive_Lymph_Nodes", data=Data_Cancer, size=8)
plt.show()
# pairwise scatter plot: Pair-Plot

sbn.set_style("whitegrid");
sbn.pairplot(Data_Cancer, hue="Survival_Status_After_5_Years", size=3);
plt.show()
# Creating Scatter plots by plotting Age and Treatment Year to put emphasis on above observation
sbn.set_style("whitegrid");
sbn.FacetGrid(Data_Cancer, hue="Survival_Status_After_5_Years", size=5) \
   .map(plt.scatter, "Age", "No_of_Positive_Lymph_Nodes") \
   .add_legend();
plt.show();
sbn.set_style("whitegrid");
sbn.FacetGrid(Data_Cancer, hue="Survival_Status_After_5_Years", size=5) \
   .map(plt.scatter, "Treatment_Year", "No_of_Positive_Lymph_Nodes") \
   .add_legend();
plt.show();