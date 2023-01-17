#impoting the necessary packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
#importing the dataset
import os
print(os.listdir("../input"))
df = pd.read_csv('../input/haberman.csv')

#Print the number of datapoints and features.
print(df.shape)
#Check the columns in the dataset.
df.columns
#add columns name to the dataset and recheck the columns names.
df.columns=["Age","Operation_year","Axil_nodes","Surv_status"]  
print(df.columns)
#Datapoint per class.
#Surv_status: 1-Survived,2-Died
df["Surv_status"] = df["Surv_status"].apply(lambda x: "Survived" if x == 1 else "Died")
df["Surv_status"].value_counts()
df.head(10) # no of Rows =10
df.describe()
# Distribution of Operation Year
sns.FacetGrid(df, hue="Surv_status", size=5)\
.map(sns.distplot, "Operation_year").add_legend();
plt.show();
# Distribution for Age of Operation 
sns.FacetGrid(df, hue="Surv_status", size=5)\
.map(sns.distplot, "Age").add_legend();
plt.show();
# Distribution for axil_ nodes of Operation 
sns.FacetGrid(df, hue="Surv_status", size=5)\
.map(sns.distplot, "Axil_nodes").add_legend();
plt.show();
Survived= df.loc[df["Surv_status"]== "Survived"]
Died = df.loc[df["Surv_status"]=="Died"]


plt.figure(figsize=(20,5))
i=1
for state in (list(df.columns)[:-1]):
#survived
    plt.subplot(1,3,i)
    Counts , bin_edges = np.histogram(Survived[state],bins=20,density=True)
    pdf=Counts/sum(Counts)
    cdf = np.cumsum(Counts)
    plt.plot(bin_edges[1:],cdf,label="cdf of survived",color="red")
    plt.plot(bin_edges[1:],pdf,label="pdf of survived",color="black")

#Death
    Counts , bin_edges = np.histogram(Died[state],bins=20,density=True)
    pdf=Counts/sum(Counts)
    cdf = np.cumsum(Counts)
    plt.plot(bin_edges[1:],cdf,label="cdf of Death")
    plt.plot(bin_edges[1:],pdf,label="pdf of Death")
    plt.xlabel(state)
    plt.grid()
    plt.legend()
    i+=1
plt.show()
# Box_plot
print("********************************* Box Plot ***********************************************")
plt.figure(figsize=(20,5))
j=1
for features in (list(df.columns)[:-1]):  
    plt.subplot(1,3,j); j+=1 
    sns.boxplot(x= 'Surv_status',y= features,data=df)
plt.grid()    
plt.show()

print("*********************************** Violin Plot ******************************************")
# violin_plot
plt.figure(figsize=(20,5))
k=1
for features in (list(df.columns)[:-1]):  
    plt.subplot(1,3,k); k+=1 
    sns.violinplot(x= 'Surv_status',y= features,data=df)
plt.grid()
plt.show()

#Pair Plot
df['Surv_status'] = df['Surv_status'].astype('category')
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Surv_status",vars = ["Age","Operation_year","Axil_nodes"], size = 3)
plt.show()