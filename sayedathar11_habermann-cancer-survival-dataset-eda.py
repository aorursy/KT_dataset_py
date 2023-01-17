import os
print(os.listdir('../input'))
import pandas as pd
df = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', header=None, names=['age', 'year', 'nodes', 'status'])
print(df.head())
df.columns
## What are the Number of Data Points ? 
df.shape[0]
## What are Number of Columns ?
df.shape[1]
#https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
df['status'] = df['status'].map({1 : 'Survived' , 2 : 'Not Survived'})
df['status'].value_counts()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_style("whitegrid");
sns.FacetGrid(df, hue="status",size=5) \
    .map(plt.scatter , "nodes" , "age") \
    .add_legend();
plt.show();
## How many Patients are Survivors Who have less than 10 nodes ?
df.query('status =="Survived" & nodes <= 10')['status'].value_counts()
## How many Patients Are Non Survivors Who have Less than 10 nodes ?
df.query('status =="Not Survived" & nodes <= 10')['status'].value_counts()
df.query('status =="Survived" & age <= 60')['status'].value_counts()
df.query('status =="Not Survived" & age <= 60')['status'].value_counts()
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="status", size=3);
plt.show()
sns.FacetGrid(df , hue = 'status' , size = 5) \
    .map(sns.distplot , 'nodes') \
    .add_legend();
plt.show()
sns.FacetGrid(df , hue = 'status' , size = 5) \
    .map(sns.distplot , 'age') \
    .add_legend();
plt.show()
sns.FacetGrid(df , hue = 'status' , size = 5) \
    .map(sns.distplot , 'year') \
    .add_legend();
plt.show()
survived = df.loc[df['status'] == 'Survived']
not_survived = df.loc[df['status'] == 'Not Survived']
## What is Mean Age of Survivor ?
print("Mean Age of Survivor is :")
print(np.mean(survived['age']))
## What is Mean Age of Non Survivor ?
print("Mean Age of Non Survivor is :")
print(np.mean(not_survived['age']))
## What is Mean Axil Nodes Count of Survivor ?
print("Mean Axil Nodes Count of Survivor is :")
print(np.mean(survived['nodes']))
## What is Mean Axil Nodes Count of Non Survivor ?
print("Mean Axil Nodes Count of Non Survivor is :")
print(np.mean(not_survived['nodes']))
## Lets Compute Median Also 
print("Median Age of Survivor is :")
print(np.median(survived['age']))
print("Median Age of Non Survivor is :")
print(np.median(not_survived['age']))
print("Median Axil Nodes Count of Survivor is :")
print(np.median(survived['nodes']))
print("Median Axil Nodes Count of Non Survivor is :")
print(np.median(not_survived['nodes']))
print("\n90th Percentiles:")
print("For Survivors")
print(np.percentile(survived["nodes"],90))
print("For Non Survivors")
print(np.percentile(not_survived["nodes"],90))

sns.boxplot(x='status',y='nodes', data=df )
plt.show()
