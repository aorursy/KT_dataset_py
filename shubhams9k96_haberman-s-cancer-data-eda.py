import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/haberman.csv')

df.columns= ['age', 'op_year', 'axil_nodes', 'survived'] 

print(df.head())

df['survived'] = df['survived'].map({1:'Yes', 2: 'No'})

df['survived'] = df['survived'].astype('category')

print(df.describe())
print(df['survived'].value_counts())

df.groupby(['survived']).mean()
df_yes= df.loc[df.survived=='Yes']

df_no= df.loc[df.survived=='No']
print("Average age of survivors: ", np.mean(df_yes.age))

print("Average age of Non survivors: ", np.mean(df_no.age))



print("Median age of survivors ", np.median(df_yes.age))

print("Median age of survivors ", np.median(df_no.age))



print("Percentile age of survivors: ", np.percentile(df_yes.age, np.arange(0,100,25)))

print("Percentile age of non survivors: ", np.percentile(df_no.age,  np.arange(0,100,25)))
print("Average #nodes of survivors: ", np.mean(df_yes.axil_nodes))

print("Average #nodes of Non survivors: ", np.mean(df_no.axil_nodes))



print("Median #nodes of survivors ", np.median(df_yes.axil_nodes))

print("Median #nodes of survivors ", np.median(df_no.axil_nodes))



print("Percentile #nodes of survivors: ", np.percentile(df_yes.axil_nodes, np.arange(0,100,25)))

print("Percentile #nodes of non survivors: ", np.percentile(df_no.axil_nodes,  np.arange(0,100,25)))



print("Percentile #nodes of survivors: ", np.percentile(df_yes.axil_nodes, 90))

print("Percentile #nodes of non survivors: ", np.percentile(df_no.axil_nodes, 90))
# fig, axes= plt.subplots(1,3, figsize=(15,5))

for idx, feature in enumerate(df.columns[:-1]):

    sns.FacetGrid(df,  hue='survived', height=5).map(sns.distplot, feature).add_legend()

    plt.ylabel('Density')

    plt.title('Probability Density function for {}'.format(feature))

    plt.show()

df.hist()

plt.figure(figsize=(15,15))

for idx, feature in enumerate(list(df.columns[:-1])):

    counts, bins= np.histogram(df[feature], bins=10, density=True)

    pdf= counts/sum(counts)

    cdf= np.cumsum(pdf)

    plt.subplot(3,3,idx+1)

    plt.plot(bins[1:], pdf, label="PDF")

    plt.plot(bins[1:], cdf, label= "CDF")

    plt.xlabel(feature)

plt.legend()    

plt.show()

    
fig, axes = plt.subplots(1,3, figsize= (15,5))

for idx, feature in enumerate(df.columns[:-1]):

    sns.boxplot(x= 'survived', y= feature, data= df, ax= axes[idx])

plt.show()    
fig, axes = plt.subplots(1,3, figsize= (15,5))

for idx, feature in enumerate(df.columns[:-1]):

    sns.violinplot(x= 'survived', y= feature, data= df, ax= axes[idx])

plt.show()    
sns.set_style('whitegrid')

sns.pairplot(df.loc[:, df.columns!='Id'], hue= 'survived', height= 5)

plt.show()