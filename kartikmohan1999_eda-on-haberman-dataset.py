# IMPORTING PACKAGES
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Importing file
df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",header=None)
df.columns = ['age', 'year', 'axillary_nodes', 'survival_status']
df.head()
print(df.shape)
df.info()
# Convert to Categorical
df['survival_status'] = df['survival_status'].map({1:'Yes', 2:'No'})
df.head() 
# Statistics
df.describe()
df["survival_status"].value_counts()
sns.set_style("whitegrid")
sns.FacetGrid(df,hue='survival_status',height=5).map(sns.distplot,'axillary_nodes').add_legend()
plt.title(' POSITIVE AXILLARY NODES')
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(df,hue='survival_status',height=5).map(sns.distplot,"age").add_legend()   
sns.set_style("whitegrid")
sns.FacetGrid(df,hue='survival_status',height=5).map(sns.distplot,"year").add_legend()   
sns.boxplot(x='survival_status',y='axillary_nodes',data=df)
plt.show()

sns.violinplot(x='survival_status',y='axillary_nodes',data=df)
plt.show()
sns.boxplot(x='survival_status',y='age',data=df)
plt.show()

sns.violinplot(x='survival_status',y='age',data=df)
plt.show()
sns.boxplot(x='survival_status',y='year',data=df)
plt.show()

sns.violinplot(x='survival_status',y='year',data=df)
plt.show()
sns.pairplot(df,hue='survival_status')
plt.show()