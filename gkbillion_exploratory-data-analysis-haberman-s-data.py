import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import os

#Load haberman.csv into a pandas dataFrame
columns = ['age','year','infected_axillary_nodes','survival_status']
haberman_df = pd.read_csv('../input/haberman.csv',names=columns)

#Verify the data that has been loaded
haberman_df.head(5)
#print the unique values of the target column

print(haberman_df['survival_status'].unique())
# The values of 'survival_status' column are not that meaningful. 
# For improved readability of the subsequent graphical
# analysis and observations we are going to map the two possible labels 1 and 2 to 
#'Survived' and 'Not Survived' respectively

haberman_df.survival_status.replace([1, 2], ['Survived', 'Not Survived'], inplace = True)
haberman_df.head(8)
#(Q) How many data points for each class or label ar present

haberman_df['survival_status'].value_counts()
# Drawing a general idea about the data from the basic statistical parameters
haberman_df.describe()
# pairwise scatter plot: Pair-Plot
# Only possible to view 2D patterns
# NOTE: the diagonal elements are PDFs for each feature.

sb.set_style("whitegrid");
sb.pairplot(haberman_df, hue="survival_status", size=4);
plt.show()
#Histogram and PDF for the feature age, year and infected axillary node

for idx, feature in enumerate(list(haberman_df.columns)[0:3]):
    sb.FacetGrid(haberman_df, hue="survival_status", size=5) \
      .map(sb.distplot, feature) \
      .add_legend();
plt.show()
# Getting seperate data for "Survived and "Not Survived" for subsequent processing of 
# data

haberman_survived = haberman_df.loc[haberman_df["survival_status"] == "Survived"]
haberman_notSurvived = haberman_df.loc[haberman_df["survival_status"] == "Not Survived"]

#Analysing data for patients who survived

plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman_df.columns)[0:3]):
    plt.subplot(1, 3, idx+1)
    print("------------------- Survived -------------------------")
    print("--------------------- "+ feature + " ----------------------------")
    counts, bin_edges = np.histogram(haberman_survived[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    
    # Compute PDF
    pdf = counts/(sum(counts))
    print("PDF:  {}".format(pdf))

    # Compute CDF
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
          
    # Plot the above cumputed values      
   # plt.subplot(1, 3, idx+1)      
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(feature)
    plt.grid()
plt.show()
#Analysing data for patients who did not survived

plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman_df.columns)[0:3]):
    plt.subplot(1, 3, idx+1)
    print("------------------- Not Survived -------------------------")
    print("--------------------- "+ feature + " ----------------------------")
    counts, bin_edges = np.histogram(haberman_notSurvived[feature], bins=10, \
                                     density=True)
    print("Bin Edges: {}".format(bin_edges))
    
    # Compute PDF
    pdf = counts/(sum(counts))
    print("PDF:  {}".format(pdf))

    # Compute CDF
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
          
    # Plot the above cumputed values      
   # plt.subplot(1, 3, idx+1)      
    plt.plot(bin_edges[1:], pdf)
    plt.plot(bin_edges[1:], cdf)
    plt.xlabel(feature)
    plt.grid()
plt.show()
#Box plot
plt.figure(figsize=(20,5))
xlabel = 'survival_status fig. ' + '(idx+1)'
for idx, feature in enumerate(list(haberman_df.columns)[0:3]):
    plt.subplot(1, 3, idx+1)
    sb.boxplot( x='survival_status', y=feature, data=haberman_df)
plt.show()  
#Violin Plot
plt.figure(figsize=(20,5))
for idx, feature in enumerate(list(haberman_df.columns)[0:3]):
    plt.subplot(1, 3, idx+1)
    sb.violinplot(x='survival_status', y=feature, data=haberman_df, size=8)
plt.show()