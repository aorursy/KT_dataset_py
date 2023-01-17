import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#Load haberman.csv into a pandas dataFrame.
df = pd.read_csv('../input/haberman.csv', \
                 header=None, \
                 names=['Age', 'Op_Year', 'axil_nodes', 'Surv_status'])
#How many data-points and features
print (df.shape)
#Different data-points for Survival status
print(list(df['Surv_status'].unique()))
# 1=> positive, 2=> negative
df["Surv_status"]=df["Surv_status"].map({1:'positive',2:'negative'})
df.head()
#Balanced dataset or not?
print (df["Surv_status"].value_counts())
print("*"*50)
print (df["Surv_status"].value_counts(normalize= True))
#Dividing the dataset into 2 datasets of positive and negative result.
positive=df.loc[df['Surv_status']=='positive']
negative=df.loc[df['Surv_status']=='negative']
print (df.describe())
#Mean and Std-Deviation
print ("Age")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["Age"])))
print ("  negative result- "+str(np.mean(negative["Age"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["Age"])))
print ("  negative result- "+str(np.std(negative["Age"])))
print ()
print("*"*50)
print ()
print ("Year of Operation")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["Op_Year"])))
print ("  negative result- "+str(np.mean(negative["Op_Year"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["Op_Year"])))
print ("  negative result- "+str(np.std(negative["Op_Year"])))
print ()
print("*"*50)
print ()
print ("No of Auxillary Nodes")
print ("  Mean:")
print ("  positive result- "+str(np.mean(positive["axil_nodes"])))
print ("  negative result- "+str(np.mean(negative["axil_nodes"])))
print ()
print ("  Standard Devation:")
print ("  positive result- "+str(np.std(positive["axil_nodes"])))
print ("  negative result- "+str(np.std(negative["axil_nodes"])))



#90th percentile
print ('90th Percentile')
print ()
print ("Age")
print ("  positive result- "+str(np.percentile(positive["Age"],90)))
print ("  negative result- "+str(np.percentile(negative["Age"],90)))
print("*"*50)
print ("Year of Operation")
print ("  positive result- "+str(np.percentile(positive["Op_Year"],90)))
print ("  negative result- "+str(np.percentile(negative["Op_Year"],90)))
print("*"*50)
print ("No of Auxillary Nodes")
print ("  positive result- "+str(np.percentile(positive["axil_nodes"],90)))
print ("  negative result- "+str(np.percentile(negative["axil_nodes"],90)))
print ("  general result- "+str(np.percentile(df["axil_nodes"],90)))


#PDF
sns.set_style("whitegrid")
for index, feature in enumerate(list(df.columns)[:-1]):
    sns.FacetGrid(df,hue='Surv_status',height=4).map(sns.distplot,feature).add_legend()
    plt.show()
#CDF
plt.figure(figsize=(20,5))
for index, feature in enumerate(list(df.columns)[:-1]):
    plt.subplot(1, 3, index+1)
    print("\n********* "+feature+" *********")
    counts, bin_edges = np.histogram(df[feature], bins=10, density=True)
    print("Bin Edges: {}".format(bin_edges))
    pdf = counts/sum(counts)
    print("PDF: {}".format(pdf))
    cdf = np.cumsum(pdf)
    print("CDF: {}".format(cdf))
    plt.plot(bin_edges[1:], pdf, bin_edges[1:], cdf)
    plt.xlabel(feature)
#Box Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.boxplot( x='Surv_status', y=feature, data=df, ax=axes[idx])
plt.show()
#Violin Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, feature in enumerate(list(df.columns)[:-1]):
    sns.violinplot( x='Surv_status', y=feature, data=df, ax=axes[idx])
plt.show()
sns.set_style("whitegrid")
sns.pairplot(df,hue='Surv_status',height=4)
plt.show()