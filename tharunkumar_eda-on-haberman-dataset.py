#Import all required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#Lets load haberman data into dataframe.
hman_data = pd.read_csv('../input/haberman.csv',
                        names=['age','year_of_op','axilary_nodes_cnt','survival_status_a5'])
# No of data points(Rows) with features(Attributes)
hman_data.shape
#Columns in our dataset, we have named it when we read into data frame, let look with below command
hman_data.columns
# Let's describe the data frame...
hman_data.describe()
# Let's change survival status attribute from 1 to "yes" and 2 to "no" to improve readability.

hman_data['survival_status_a5'] = hman_data['survival_status_a5'].map({1: 'Yes' , 2 : 'No'})

#No of data points per class
hman_data['survival_status_a5'].value_counts()
# Let's see the data types of data frame features.
hman_data.dtypes
%matplotlib inline
hman_data.plot(kind='scatter', x = 'age',y = 'axilary_nodes_cnt')
# let's see with seaborn where we can color each survival status
sns.set_style("whitegrid")
sns.FacetGrid(hman_data, hue = 'survival_status_a5', height = 5)\
   .map(plt.scatter,"age", "axilary_nodes_cnt")\
   .add_legend();
plt.show()
sns.pairplot(hman_data,hue = "survival_status_a5", height = 3)
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(hman_data.axilary_nodes_cnt, hue=hman_data['survival_status_a5'])
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(hman_data.age, hue=hman_data['survival_status_a5'])
plt.legend(loc='upper right')
sns.countplot(hman_data.year_of_op, hue=hman_data['survival_status_a5'])
plt.legend(loc='upper right')
sns.set_style("whitegrid")
sns.FacetGrid(hman_data,hue="survival_status_a5", height=6)\
   .map(sns.distplot,"axilary_nodes_cnt")\
   .add_legend()
sns.set_style("whitegrid")
sns.FacetGrid(hman_data,hue="survival_status_a5", height=6)\
   .map(sns.distplot,"age")\
   .add_legend()
sns.set_style("whitegrid")
sns.FacetGrid(hman_data,hue="survival_status_a5", height=6)\
   .map(sns.distplot,"year_of_op")\
   .add_legend()
# Let's see with CDF's
for i in ['axilary_nodes_cnt', 'age', 'year_of_op']:
    
   hman_data_s = hman_data.loc[hman_data["survival_status_a5"] == "Yes"];   
   counts, bin_edges = np.histogram(hman_data_s[i], bins=10, 
                                 density = True)
   pdf = counts/(sum(counts))
   cdf = np.cumsum(pdf)
   plt.xlabel(i)
   plt.plot(bin_edges[1:],pdf,label = 'PDF_YES')
   plt.plot(bin_edges[1:], cdf, label = 'CDF_YES')
   
   hman_data_n = hman_data.loc[hman_data["survival_status_a5"] == "No"]; 
   counts, bin_edges = np.histogram(hman_data_n[i], bins=10, 
                                    density = True)
   pdf = counts/(sum(counts))
   cdf = np.cumsum(pdf)
   plt.xlabel(i)
   plt.plot(bin_edges[1:],pdf, label = 'PDF_NO')
   plt.plot(bin_edges[1:], cdf, label = 'CDF_NO')
   plt.legend()
   plt.show()
sns.boxplot(x='survival_status_a5',y='axilary_nodes_cnt',data=hman_data)
sns.boxplot(x='survival_status_a5',y='year_of_op',data=hman_data)
sns.boxplot(x='survival_status_a5',y='year_of_op',data=hman_data)
sns.violinplot(x='survival_status_a5',y='axilary_nodes_cnt',data=hman_data)
sns.violinplot(x='survival_status_a5',y='year_of_op',data=hman_data)
sns.violinplot(x='survival_status_a5',y='age',data=hman_data)