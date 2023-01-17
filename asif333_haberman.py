import numpy as np 

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



hb=pd.read_csv("../input/haberman.csv",)

print(hb.shape)
print(hb.columns)
hb["status"].value_counts()
hb.head()
print("total people survied more than 5 years= ",f'{(225/306)*100}')
surv_more = hb[hb['status']==1]

surv_more.describe()
surv_less = hb[hb['status']==2]

surv_less.describe()
sns.FacetGrid(hb, hue = 'status' , height = 5).map(sns.distplot , 'operation_year').add_legend();

plt.show();
sns.FacetGrid(hb , hue = 'status' , height=5).map(sns.distplot , 'age').add_legend();

plt.show()
sns.FacetGrid(hb, hue = 'status' , size =5).map(sns.distplot , 'axil_nodes').add_legend();

plt.show()
sns.set_style("whitegrid");

sns.FacetGrid(hb, hue = 'status' , size =5).map(plt.scatter , 'operation_year','axil_nodes').add_legend();

plt.show()
plt.close();

sns.set_style("whitegrid");

sns.pairplot(hb,hue='status',height=3);

plt.show()
sns.boxplot(x = 'status',y = 'operation_year',data = hb)

plt.show()
sns.boxplot(x = 'status',y = 'age',data = hb)

plt.show()
sns.boxplot(x = 'status',y = 'axil_nodes',data = hb)

plt.show()
sns.violinplot(x="status", y="operation_year", data=hb, size=8)

plt.show()
sns.violinplot(x="status", y="axil_nodes", data=hb, size=8)

plt.show()
sns.jointplot(x='axil_nodes' , y = 'age' , data = hb , kind = 'kde')

plt.show()
plt.close()

sns.jointplot(x='operation_year' , y='axil_nodes' , data = hb , kind='kde')

plt.show()