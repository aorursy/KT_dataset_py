# check for the input dataset
import os
print(os.listdir('../input'))
#importing essentials libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
#Load Haberman data into a pandas dataFrame.
haberman = pd.read_csv("../input/haberman.csv", header=None,names=['Age','Op_Year','axil_nodes_det','Surv_status'])
#checking the first 5 values of the dataframe
haberman.head()
# how many data-points and features?
print(haberman.shape)
#What are the column names in our dataset?
print(haberman.columns)
#How many data points for each class are present? 
haberman["Surv_status"].value_counts()
#2-D scatter plot:
haberman.plot(kind='scatter', x='Age', y='Op_Year', title='2-D scatter plot Age Vs Op_Year') ;
plt.show()
#this plot doesn't give any information about the classes
# 2-D Scatter plot with color-coding for each class.

sns.set_style("whitegrid");
sns.FacetGrid(haberman, hue="Surv_status", height=4) \
   .map(plt.scatter, "Age", "Op_Year") \
   .add_legend();
plt.title('2-D scatter plot Age Vs Op_Year')
plt.show();
#this pair plot gives class dependancies w.r.t. each variable/feature
plt.close();
sns.set_style("whitegrid");
sns.pairplot(haberman, hue="Surv_status",vars=["Age", "Op_Year","axil_nodes_det"], height=5);
plt.suptitle('Pair Plot for each feature')
plt.show()
#1-D scatter plot of AGE
haberman_one = haberman.loc[haberman["Surv_status"] == 1];
haberman_two= haberman.loc[haberman["Surv_status"] == 2];

plt.plot(haberman_one["Age"], np.zeros_like(haberman_one['Age']), 'o')
plt.plot(haberman_two["Age"], np.zeros_like(haberman_two['Age']), 'o')
plt.legend('12')
plt.xlabel("Age")
plt.title('1-D scatter plot for AGE')
plt.show()
# Histogram and PDF of AGE for both classes
sns.FacetGrid(haberman, hue="Surv_status", height=5) \
   .map(sns.distplot, "Age") \
       .add_legend();
plt.title('Histogram and PDF of AGE')
plt.ylabel('count')
plt.show();
# Histogram and PDF of Op_Year for both classes
sns.FacetGrid(haberman, hue="Surv_status", height=5) \
   .map(sns.distplot, "Op_Year") \
       .add_legend();
plt.title('Histogram and PDF of Op_Year')
plt.ylabel('count')
plt.show();
# CDF plot of Age for class 1
counts, bin_edges = np.histogram(haberman_one['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges);
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf);
plt.plot(bin_edges[1:], cdf)
plt.legend('1')
plt.xlabel("Age")
plt.ylabel("Probability")
plt.title("CDF plot of Age for class 1")
plt.show();
#  CDF plot of Age for both classes

counts, bin_edges = np.histogram(haberman_one['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)



counts, bin_edges = np.histogram(haberman_two['Age'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.xlabel("Age")
blue_patch = mpatches.Patch(color='blue', label='PDF class 1')
green_patch = mpatches.Patch(color='green', label='PDF class 2')
orange_patch = mpatches.Patch(color='orange', label='CDF class 1')
red_patch = mpatches.Patch(color='red', label='CDF class 2')
plt.legend(handles=[blue_patch,green_patch,orange_patch,red_patch])
plt.ylabel("probability")
plt.title("CDF and PDF plot of both classes w.r.t Age")
plt.show();
#Mean, Variance, Standard-dev,  
print("mean:")
print('mean of class 1 is:',np.mean(haberman_one["Age"]))
#Mean with an outlier.
print('mean with outlier is:',np.mean(np.append(haberman_one["Age"],2240)));
print('mean of class 2 is: ',np.mean(haberman_two["Age"]))

print("\nStandard-dev:");
print('STD of class 1 is:',np.std(haberman_one["Age"]))
print('STD of class 2 is:',np.std(haberman_two["Age"]))
#Median, Quantiles, Percentiles, IQR.
print("\nmedians:")
print('median of class 1 is:',np.median(haberman_one["Age"]))
#Median with an outlier
print('median with outlier is:',np.median(np.append(haberman_one["Age"],2240)));
print('median of class 2 is:',np.median(haberman_two["Age"]))

print("\nQuantiles:")
print(np.percentile(haberman_one["Age"],np.arange(0, 100, 25)))
print(np.percentile(haberman_two["Age"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(haberman_one["Age"],90))
print(np.percentile(haberman_two["Age"],90))

print("\n85th Percentiles:")
print(np.percentile(haberman_one["Age"],85))
print(np.percentile(haberman_two["Age"],85))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(haberman_one["Age"]))
print(robust.mad(haberman_two["Age"]))
sns.boxplot(x='Surv_status',y='Age', data=haberman).set_title('Box plot of AGE and survival status')
blue_patch = mpatches.Patch(color='blue', label='class 1')
orange_patch = mpatches.Patch(color='orange', label='class 2')
plt.legend(handles=[blue_patch,orange_patch],loc=1)
plt.show()
sns.violinplot(x="Surv_status", y="Age", data=haberman, size=8)
plt.title('Violin plt of AGE and Survival status')
blue_patch = mpatches.Patch(color='blue', label='class 1')
orange_patch = mpatches.Patch(color='orange', label='class 2')
plt.legend(handles=[blue_patch,orange_patch],loc=1)
plt.show()
#2D Density plot, contors-plot
sns.jointplot(x="Age", y="Op_Year", data=haberman_one, kind="kde");
plt.suptitle('Contors plot of AGE and Op_Year')
plt.show();