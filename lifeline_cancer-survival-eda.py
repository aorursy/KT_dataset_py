# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from termcolor import colored

warnings.filterwarnings("ignore")
#import hibernann data set first
habermans_ds = pd.read_csv("../input/haberman.csv")
#what are the columns
print("columns ",habermans_ds.columns)
print(colored("###### Explore dimension of data ######",'blue'))
print("rows ",habermans_ds.shape[0], "  columns ",habermans_ds.shape[1])
#print(hebermans_ds.describe)
print(habermans_ds.size)
head  = pd.DataFrame.head(habermans_ds)
tail  = pd.DataFrame.tail(habermans_ds)
print(head)
print("###########################")
print(tail)
print(colored("#######Print first 10 rows #######","cyan"))
print(habermans_ds.head(10))
print(colored("#######Print last 10 rows ######","cyan"))
data_set_temp = habermans_ds.head(10)
print(data_set_temp)
#print(habermans_ds.describe)
print(habermans_ds.info)
# Convert survival status 1 to Yes and 2 to no
habermans_ds['survival_status'] = habermans_ds['survival_status'].map({1:True, 2:False})
habermans_ds['survival_status'] = habermans_ds['survival_status'].astype('category')
print(habermans_ds.head())

print(colored("What is the survival Percentage?","cyan"))
grouped = habermans_ds.groupby('survival_status')
survived_group = grouped.get_group(True)
not_survived_group = grouped.get_group(False)
print("sruvived ",len(survived_group))
print("NOT sruvived ",len(not_survived_group))
#for key, item in grouped:
 #   print(grouped.get_group(key), "\n\n")
total_rows = habermans_ds.count()[0]
survival_percentage = 100*(len(survived_group)/total_rows)
print("Sruvival Percentage : ",survival_percentage)
# Any results you write to the current directory are saved as output.
habermans_ds.plot(kind='scatter', y='positive_axillary_nodes', x='Age') ;
plt.show()
#Can't understand much except Number of positive axillary nodes detected are majority <10
# 2-D Scatter plot with color-coding for each flower type/class.
# Here 'sns' corresponds to seaborn. 
sns.set_style('whitegrid')
sns.FacetGrid(habermans_ds,hue="survival_status", height=5) \
   .map(plt.scatter, "Age","positive_axillary_nodes") \
   .add_legend();
plt.show();
nodes_det_lessthan_10 = habermans_ds[habermans_ds.positive_axillary_nodes <= 6]
grouped_as_surv = nodes_det_lessthan_10.groupby('survival_status')
survived_group = grouped_as_surv.get_group(True)
not_survived_group = grouped_as_surv.get_group(False)
total_rows = nodes_det_lessthan_10.count()[0]
survival_percentage = 100*(len(survived_group)/total_rows)
print("Suvival percentage when axile nodes <=6 ",survival_percentage)

plt.plot(survived_group["Age"],np.zeros_like(survived_group["Age"]),'c')
plt.plot(not_survived_group["Age"],np.zeros_like(not_survived_group["Age"]),'r')
plt.show()

print(colored("Suvival for age 30 to 35 yrs are looks very high ",'cyan'))

#survived_group = habermans_ds[habermans_ds.Surv_status == True]
#not_survived_group = habermans_ds[habermans_ds.Surv_status == False]
#plt.plot(survived_group["axil_nodes_det"],np.zeros_like(survived_group["axil_nodes_det"]),'c')
#plt.plot(not_survived_group["axil_nodes_det"],np.zeros_like(not_survived_group["axil_nodes_det"]),'r')
#plt.show()
sns.FacetGrid(habermans_ds, hue="survival_status", height=5) \
   .map(sns.distplot, "positive_axillary_nodes") \
   .add_legend();

plt.show()
print(colored("Survival high when axile nodes <6 ",'cyan'))

sns.FacetGrid(habermans_ds, hue="survival_status", height=5) \
   .map(sns.distplot, "year") \
   .add_legend();

plt.show()

print(colored("Survival is high if operated between 1963 and 1967 ",'cyan'))

sns.FacetGrid(habermans_ds, hue="survival_status", height=5) \
   .map(sns.distplot, "Age") \
   .add_legend();

plt.show()

print(colored("Suvival high between the age  group of 53 to 56 ",'cyan'))
plt.close();
sns.set_style("ticks");
#Delete the survial_status column it's of no use
haberman_ds_for_pairplot = habermans_ds.drop('survival_status', axis=1)
sns.pairplot(haberman_ds_for_pairplot, hue="positive_axillary_nodes", height=4);
plt.show()
counts, bin_edges = np.histogram(habermans_ds['positive_axillary_nodes'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
plt.plot(bin_edges[1:],pdf);
cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],cdf)
plt.show()

survived_group = habermans_ds[habermans_ds.survival_status == True]
not_survived_group = habermans_ds[habermans_ds.survival_status == False]
print("Means:")
print(np.mean(survived_group["Age"]))
#Mean with an outlier.
print(np.mean(np.append(survived_group["Age"],50)));
print(np.mean(not_survived_group["Age"]))


print("\nStd-dev:");
print(np.std(survived_group["Age"]))
print(np.std(not_survived_group["Age"]))

print("Means for positive_axillary_nodes:")
print(np.mean(survived_group["positive_axillary_nodes"]))
#Mean with an outlier.
print(np.mean(np.append(survived_group["positive_axillary_nodes"],20)));
print(np.mean(not_survived_group["positive_axillary_nodes"]))


print("\nStd-dev for positive_axillary_nodes:");
print(np.std(survived_group["positive_axillary_nodes"]))
print(np.std(not_survived_group["positive_axillary_nodes"]))
print("\nMedians:")
print(np.median(survived_group["positive_axillary_nodes"]))
#Median with an outlier
print(np.median(np.append(survived_group["positive_axillary_nodes"],20)));
print(np.median(survived_group["positive_axillary_nodes"]))
print(np.median(not_survived_group["positive_axillary_nodes"]))


print("\nQuantiles:")
print(np.percentile(survived_group["positive_axillary_nodes"],np.arange(0, 100, 25)))
print(np.percentile(not_survived_group["positive_axillary_nodes"],np.arange(0, 100, 25)))

print("\n90th Percentiles:")
print(np.percentile(survived_group["positive_axillary_nodes"],90))
print(np.percentile(not_survived_group["positive_axillary_nodes"],90))


from statsmodels import robust
print ("\nMedian Absolute Deviation")
print(robust.mad(survived_group["positive_axillary_nodes"]))
print(robust.mad(not_survived_group["positive_axillary_nodes"]))
sns.boxplot(x='survival_status',y='Age', data=survived_group)
sns.boxplot(x='survival_status',y='Age', data=not_survived_group)
plt.show()

sns.boxplot(x='survival_status',y='positive_axillary_nodes', data=survived_group)
sns.boxplot(x='survival_status',y='positive_axillary_nodes', data=not_survived_group)
plt.show()