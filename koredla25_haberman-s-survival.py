# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
# reading the dataset
df=pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv')
# column names 
df.columns
# assigning the column names
df.columns=['age','op_year','axil_nodes','surv_status']
# column names
df.columns
# first 5 data points in the dataset
df.head(5)
# shape of the data set 
df.shape
# survived status assign boolien values to 1 for true 2 for false 
df['surv_status']=df['surv_status'].map({1:True,2:False})
# just checking the first 5 data points
df.head(5)
df.shape
df.value_counts()
# we can see how many survived and non survived 
df['surv_status'].value_counts()
df['axil_nodes'].value_counts()
df['age'].value_counts()
df['op_year'].value_counts()
# 2 D scatter Plot
df.plot(kind='scatter',x='age',y='axil_nodes')
plt.show()
sns.set_style('whitegrid')
sns.FacetGrid(df,hue='surv_status',height=4).map(plt.scatter,'age','axil_nodes').add_legend()
plt.show()
sns.set_style('whitegrid')
sns.pairplot(df,hue='surv_status',height=3)
plt.show()

sns.FacetGrid(df,hue='surv_status',height=5).map(sns.distplot, 'axil_nodes').add_legend()
plt.show()
sns.FacetGrid(df,hue='surv_status',height=5).map(sns.distplot,'op_year').add_legend()
plt.show()
sns.FacetGrid(df, hue='surv_status', height=5).map(sns.distplot, 'age').add_legend()
plt.show()
sns.FacetGrid(df, hue='surv_status',size=5).map(sns.distplot,'op_year').add_legend()
plt.show()
counts,bin_edges=np.histogram(df['axil_nodes'],bins=20,density = True)
pdf=counts/(sum(counts))
print(pdf)
print(bin_edges)

cdf=np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)

plt.xlabel('axil_nodes')
plt.show()




sns.boxplot(x='surv_status', y= 'axil_nodes' , data =df)
plt.show()
sns.violinplot(x='surv_status',y='axil_nodes', data=df,size=9)
plt.show()
