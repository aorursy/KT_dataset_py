import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore") #just imported so that no warning is showed 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("/kaggle/input/habermans-survival-data-set/haberman.csv") #reading the csv file as a dataframe

data.head() #displaying the first five records
names=['age','operated_year','axil_nodes','survival_status']

data.columns=names
data.shape
data.columns
data.info()
data['survival_status'].unique() #get unique values
pd.value_counts(data.survival_status) #get count of unique values present in the survival column
#X-axis label denotes the value on the x axis corresponding to it and Y-axis is for the frequency

for i in (data.columns)[:-1]:

    fig = sns.FacetGrid(data, hue='survival_status', size=5)

    fig.map(sns.distplot, i).add_legend()

    plt.show()
#2D scatter plot 

plt.close();

sns.set_style("whitegrid")

sns.pairplot(data, hue="survival_status",vars=['age', 'operated_year', 'axil_nodes'],height=5) #hue is color it by which label

plt.show()
for i in (data.columns)[:-1]:

    sns.boxplot(x='survival_status',y=i,data=data)

    plt.show()
#Violin plot is the combination of probability density function(PDF) and box plot.

for i in (data.columns)[:-1]:

    sns.violinplot(x="survival_status",y=i,data=data,size=8)

    plt.show()
plt.figure(figsize=(10,5))

sns.heatmap(data.corr(),annot=True,linewidth=0.5,cmap='coolwarm')