import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


data = pd.read_csv('../input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv')
data.head()
data.shape
data.columns
data.info()

# Different types of species we have
# Number of data points we have for each species.
print("Number of data points in each species")
data['species'].value_counts().plot(kind = 'bar')
data['species'].value_counts()
# Checking the percentage of missing values
print("Missing Values")
100*data.isnull().sum()/len(data)
col = ['culmen_depth_mm','culmen_length_mm','flipper_length_mm','body_mass_g']
for column in col:
    data[column].fillna(data[column].median(),inplace = True)
data['sex'] = data['sex'].fillna('MALE')
sns.set_style('whitegrid')
sns.FacetGrid(data, hue ="species", size =4)\
   .map(plt.scatter,"culmen_length_mm","culmen_depth_mm")\
   .add_legend();
plt.show()
plt.close();
sns.set_style("whitegrid");
sns.pairplot(data, hue='species', size=4);
plt.show()
sns.boxplot(x = 'species', y='culmen_length_mm', data = data)
plt.show()
sns.violinplot(x='species', y = 'culmen_depth_mm', data= data)
plt.show()
sns.swarmplot(x='species', y='flipper_length_mm', data = data)
plt.show()
sns.FacetGrid(data, hue="species", height=6,)\
   .map(sns.kdeplot, "body_mass_g",shade=True)\
   .add_legend()
plt.show()
sns.jointplot(x="culmen_length_mm", y="flipper_length_mm",data = data, kind="kde", height=7, space=0)
sns.catplot(x="species", y="culmen_depth_mm", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
data[data['sex']=='.']
sns.catplot(x="island", y="culmen_length_mm", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
data.loc[336,'sex'] = 'MALE'
sns.catplot(x="species", y="culmen_length_mm", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
sns.catplot(x="species", y="culmen_depth_mm", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
sns.catplot(x="species", y="flipper_length_mm", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
sns.catplot(x="species", y="body_mass_g", hue="sex", data=data,
                height=6, kind="bar", palette="muted")
fig = sns.barplot(data= data['island'].value_counts().reset_index(), x='island', y='index')
fig.set(xlabel='', ylabel='ISLANDS')
plt.show()
# Total number of species 
data.species.value_counts()
df = data[data.island=='Biscoe']
print(df.species.value_counts())
df.species.value_counts().plot(kind='bar')
df = data[data.island=='Dream']
print(df.species.value_counts())
df.species.value_counts().plot(kind='bar')
df = data[data.island=='Torgersen']
print(df.species.value_counts())
df.species.value_counts().plot(kind='bar')
