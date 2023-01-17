import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('../input/haberman.csv')
print(df.shape)
print(df.columns)   # there are 4 data points

df=pd.read_csv('../input/haberman.csv',names=[ 'Age','Op_Year','axil_nodes_det','Surv_status'])
df['Surv_status'].value_counts()
df.info()  #The data set has no missing values.
print('Mean of age of people who survived more than 5 yaers is '+str(np.mean(df[df['Surv_status']==1]['Age'])))
print("Mean of age of people who can't survived more than 5 yaers is "+str(np.mean(df[df['Surv_status']==2]['Age'])))
df.describe()
sns.FacetGrid(df,hue='axil_nodes_det',size=5) \
    .map(sns.distplot,'Age').add_legend()
plt.show()
# for 1st objective
sns.FacetGrid(df,hue='Surv_status',size=5) \
    .map(sns.distplot,'Age').add_legend()
plt.show()
# for 2nd objective
sns.FacetGrid(df,hue='Surv_status',size=5) \
    .map(sns.distplot,'axil_nodes_det').add_legend()    # less auxilary nodes detected have high survival rate
plt.show()
#for 3rd objective
# let us make survival status with 1 as survived and 2 as not survived for simplifying data visualiztion

df_survived = df[df['Surv_status']==1]
df_not_survived = df[df['Surv_status']==2]
j=1
cols=df.columns[:3]
# making a loop for plotting histogram for 3 columns and making pdf,cdf plots for both survival and who has not survived so we can compare and draw conclusions
for i in list(cols):
    plt.figure(figsize=(10,10))
    print(str(j)+"\t"+i)
    plt.subplot(3,1,j)
    counts, bin_edges = np.histogram(df_survived[i], bins=20,density = True)
    pdf = counts/(sum(counts))
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,color='orange',label='pdf of survived')
    plt.plot(bin_edges[1:], cdf,color='red',label='cdf of survived')
    
    counts, bin_edges = np.histogram(df_not_survived[i], bins=20,density = True)
    pdf = counts/(sum(counts))
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:],pdf,color='black',label='pdf of not survived')
    plt.plot(bin_edges[1:], cdf,color='blue',label='cdf of not survived')
    
    
    plt.grid()
    plt.legend()
    plt.show()
    j=j+1
print("\n")
print(df_survived.min())  
print("\n")
print(df_not_survived.min())   
print("\n")
print(df_not_survived.max())   
print("\n")
print(df_survived.max())   
#Median, Quantiles, Percentiles, IQR.
from statsmodels import robust

cols=df.columns
print('Survived \n')
for col in cols:
    print(col)
    print("Medians:"+str(np.median(df_survived[col])))
    print("Quantiles:"+str(np.percentile(df_survived[col],np.arange(0, 100, 25))))
    print("90th Percentiles:"+str(np.percentile(df_survived[col],90)))
    print ("Median Absolute Deviation:"+str(robust.mad(df_survived[col])))
    print()


cols=df.columns
print('Not Survived \n')
for col in cols:
    print(col)
    print("Medians:"+str(np.median(df_not_survived[col])))
    print("Quantiles:"+str(np.percentile(df_not_survived[col],np.arange(0, 100, 25))))
    print("90th Percentiles:"+str(np.percentile(df_not_survived[col],90)))
    print ("Median Absolute Deviation:"+str(robust.mad(df_not_survived[col])))
    print()
sns.boxplot(x='Surv_status',y='axil_nodes_det', data=df)
plt.show()

plt.figure(figsize=(30,5))
plt.subplot(1,2,1)
sns.boxplot(x='axil_nodes_det',y='Age',hue='Surv_status', data=df)
plt.subplot(1,2,2)
sns.boxplot(x='Surv_status',y='Age', data=df)
plt.show()
j=1
for col in cols[:3]:
    plt.plot(2,2,j)
    sns.violinplot(x='Surv_status',y=col, data=df)
    plt.show()
    j=j+1
df.plot(kind='scatter', x='Age', y='axil_nodes_det') ;
plt.show()
sns.set_style("whitegrid");
# coloring the survival status to get the idea of survival status
sns.FacetGrid(df, hue="Surv_status", size=4) \
   .map(plt.scatter, "Age", "axil_nodes_det") \
   .add_legend();
plt.show();
sns.set_style("whitegrid");
sns.FacetGrid(df, hue="Surv_status", size=4) \
   .map(plt.scatter, "Age", "Op_Year") \
   .add_legend();
plt.show();
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Surv_status", size=3);
plt.show()
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="axil_nodes_det",size=4);
plt.show()
j=1
for col in cols:
    if col is 'Age':
        continue
    sns.jointplot(x="Age", y=col, data=df, kind="kde");
    plt.show()
    j=j+1
plt.show()