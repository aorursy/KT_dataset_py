import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
haberman=pd.read_csv("../input/haberman.csv")
haberman.head()
#Adding column name to data set
columns=['Age','Operation_Year','AuxillaryNode','Survival']
haberman_data=pd.read_csv('../input/haberman.csv',names=columns)
haberman_data.head()
haberman_data.shape
haberman_data.columns
haberman_data.info()
haberman_data.Survival.value_counts()
haberman_data.plot(kind='scatter',x='Age',y='AuxillaryNode')
plt.show()
sns.pairplot(haberman_data,hue="Survival")
plt.show()
sns.set_style("whitegrid")
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(plt.scatter,"Age","AuxillaryNode").add_legend()
plt.show()
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"AuxillaryNode").add_legend()
plt.show()
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"Age").add_legend()
plt.show()
sns.FacetGrid(haberman_data,hue='Survival',size=5).map(sns.distplot,"Operation_Year").add_legend()
plt.show()
#In order find Age of people who survived we need to find mean age of people
print("Mean age of patients survived:", round(np.mean(haberman_data[haberman_data['Survival'] == 1]['Age'])))
print("Mean age of patients not survived:", round(np.mean(haberman_data[haberman_data['Survival'] == 2]['Age'])))
survived=haberman_data.loc[haberman_data["Survival"]==1]
notsurvived=haberman_data.loc[haberman_data["Survival"]==2]
counts, bin_edges = np.histogram(survived['AuxillaryNode'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf  Survived',
            'Cdf  Survived'])
plt.show()
counts, bin_edges = np.histogram(notsurvived['AuxillaryNode'], bins=20, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.legend(['Pdf Died',
            'Cdf Died'])
plt.show()
haberman_data[haberman_data['Survival']==1].describe()
haberman_data[haberman_data['Survival']==2].describe()
sns.boxplot(x='Survival',y='AuxillaryNode', data=haberman_data)
plt.show()
sns.boxplot(x='Survival',y='Age', data=haberman_data)
plt.show()
sns.boxplot(x='Survival',y='Operation_Year', data=haberman_data)
plt.show()
sns.violinplot(x='Survival',y='AuxillaryNode', data=haberman_data)
plt.show()
sns.violinplot(x='Survival',y='Age', data=haberman_data)
plt.show()
sns.violinplot(x='Survival',y='Operation_Year', data=haberman_data)
plt.show()
sns.jointplot(x="Age", y="Operation_Year", data=haberman_data, kind="kde");
plt.show();