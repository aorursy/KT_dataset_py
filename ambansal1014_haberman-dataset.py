import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
haberman = pd.read_csv('../input/habermans-survival-data-set/haberman.csv', names = ['age','operation_year','axil_nodes','survived_status'])

haberman.head(5)
haberman.shape
haberman.columns
haberman["survived_status"].value_counts()
plt.scatter(haberman["age"],haberman["operation_year"],color='r')

plt.xlabel("age")

plt.ylabel("operation_year")

plt.title("Age Vs Operation_year")

plt.show()
plt.scatter(haberman["age"],haberman["axil_nodes"],color='b')

plt.xlabel("age")

plt.ylabel("axil_nodes")

plt.title("Age Vs axil_nodes")

plt.show()
plt.scatter(haberman["axil_nodes"],haberman["operation_year"],color='g')

plt.xlabel("axil_nodes")

plt.ylabel("operation_year")

plt.title("axil_nodes Vs operation_year")

plt.show()
sns.set_style('whitegrid')

sns.pairplot(haberman,hue='survived_status',height=3)

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(plt.scatter,'age','operation_year').add_legend()

plt.title("Age Vs Operation_year")

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(plt.scatter,'age','axil_nodes').add_legend()

plt.title("Age Vs axil_nodes")

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(plt.scatter,'axil_nodes','operation_year').add_legend()

plt.title("axil_nodes Vs operation_year")

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(sns.distplot,'age').add_legend()

plt.title("Age")

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(sns.distplot,'operation_year').add_legend()

plt.title("operation_year")

plt.show()
sns.set_style('whitegrid')

sns.FacetGrid(haberman,hue ='survived_status',height=5).map(sns.distplot,'axil_nodes').add_legend()

plt.title("axil_nodes")

plt.show()
print("Mean of age of people whose survived_status is 1 =",np.mean(haberman.age[haberman['survived_status']==1]))

print("Mean of age of people whose survived_status is 2 =",np.mean(haberman.age[haberman['survived_status']==2]))

print("Mean of axial_node of people whose survived_status is 1 =",np.mean(haberman.axil_nodes[haberman['survived_status']==1]))

print("Mean of axial_node of people whose survived_status is 2 =",np.mean(haberman.axil_nodes[haberman['survived_status']==2]))

print("Standard Deviation of age of people whose survived_status is 1 =",np.std(haberman.age[haberman['survived_status']==1]))

print("Standard Deviation of age of people whose survived_status is 2 =",np.std(haberman.age[haberman['survived_status']==2]))

print("Standard Deviation of axial_node of people whose survived_status is 1 =",np.std(haberman.axil_nodes[haberman['survived_status']==1]))

print("Standard Deviation of axial_node of people whose survived_status is 2 =",np.std(haberman.axil_nodes[haberman['survived_status']==2]))

print("90 percentile of age of people whose survived_status is 1 =",np.percentile(haberman.age[haberman['survived_status']==1],90))

print("90 percentile of age of people whose survived_status is 2 =",np.percentile(haberman.age[haberman['survived_status']==2],90))

print("90 percentile of axial_node of people whose survived_status is 1 =",np.percentile(haberman.axil_nodes[haberman['survived_status']==1],90))

print("90 percentile of axial_node of people whose survived_status is 2 =",np.percentile(haberman.axil_nodes[haberman['survived_status']==2],90))
sns.set_style('whitegrid')

sns.boxplot(x='survived_status',y='age',data=haberman)

plt.title("Box Plot for Age")

plt.show()
sns.set_style('whitegrid')

sns.boxplot(x='survived_status',y='operation_year',data=haberman)

plt.title("Box Plot for operation_year")

plt.show()
sns.set_style('whitegrid')

sns.boxplot(x='survived_status',y='axil_nodes',data=haberman)

plt.title("Box Plot for axil_nodes")

plt.show()
sns.set_style('whitegrid')

sns.violinplot(x='survived_status',y='age',data=haberman)

plt.title("Violin Plot for Age")

plt.show()
sns.set_style('whitegrid')

sns.violinplot(x='survived_status',y='operation_year',data=haberman)

plt.title("Violin Plot for operation_year")

plt.show()
sns.set_style('whitegrid')

sns.violinplot(x='survived_status',y='axil_nodes',data=haberman)

plt.title("Violin Plot for axil_nodes")

plt.show()