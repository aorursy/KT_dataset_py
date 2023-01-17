# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/pokemon/Pokemon.csv')
df.head(3)
df.info()
df.describe()
df.shape
df.columns
columns = df.columns.tolist()
columns[0] = 'id'
df.columns = columns
df.columns
df.isnull().sum()
df['Name'][0:5]

##Read more than one columns
df[['Name', 'Attack','Type 2']][0:4]
df_tmp = df[['Name', 'Type 1', 'Attack']]
df_tmp.head(2)
df.iloc[1]
df.iloc[3,2]
# for index, row in df.iterrows():
#     ##print(index, row)
#     print(index, row['Name'])
df.loc[df['Type 1'] == 'Fire'][0:3]
##Sort dataframe in ascending order(default) through a particular column.
df.sort_values('Attack')

##Sort dataframe in descending order.
df.sort_values('Attack', ascending =False)

##Sort dataframe using multiple columns with different order
df.sort_values(['Attack', 'Defense'], ascending =(0,1))[:5]
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] +df['Sp. Def'] +df['Speed']
df.head(3)

##Add new column using iloc[row, column].sum(axis = 1 for horizontal sum & 0 for vertical sum) 
df['Total_iloc_use'] = df.iloc[:, 4:10].sum(axis = 1)
df.head(3)
df = df.drop(columns =['Total_iloc_use'])
df[:3]
##Save your datsframe as new csv file without index
df.to_csv('modified_new.csv', index=False)

##Save in excel format i.e xlsx
df.to_excel('modified_new_excel.xlsx', index=False)
df.loc[(df['Type 1'] =='Grass') & (df['Type 2']=='Poison') & (df['Attack'] >=69)][:4]

##Save this filtered data in new dataframe
new_df = df.loc[(df['Type 1'] =='Grass') & (df['Type 2']=='Poison') & (df['Attack'] >=69)][:4]

##new_df.head(3)

##Reset index of new dataframe
new_df.reset_index(drop =True, inplace=True)
new_df.head(3)
##Find value in 'Name' column which contains mega word

df.loc[df['Name'].str.contains('Mega')][:4]

##Remove those Name from data which contains mega word  

df.loc[~df['Name'].str.contains('Mega')].head(3)
import re

##Find 'Name' which starts with 'pi' in Name column

df.loc[df['Name'].str.contains('^pi[a-z]*', flags =re.I, regex= True)].head(4)
##Changing the value 'Fire' to 'Flamer' in the 'Name' column 

df.loc[df['Type 1'] =='Fire', 'Type 1'] ='Flamer'
df

##Make all fire type pokemon legendery
df.loc[df['Type 1'] =='Flamer', 'Legendary'] =True
df

##Add new column and assign true value whose attack value is greater than 69
df.loc[df['Attack'] > 69, 'Beast'] = True
df.loc[df['Attack'] < 69, 'Beast'] = False
df.head(5)
df_mod = pd.read_csv('modified_new.csv')
df_mod.head(3)
##Aggregate Statistics will be done with groupby(), sum(), count() and mean() function.

##Use groupby function to find mean of all Type 1 pokemon and sort in ascending order w.r.t Attack.

df_mod.groupby(['Type 1']).mean().sort_values('Attack', ascending=False)

##Sum up all Type 1 pokemon features
df_mod.groupby(['Type 1']).sum()

##Count number of Type 1 pokemon in dataframe
df_mod.groupby(['Type 1']).count()

##Make another column count to count Type 1 pokemon more efficiently &  it is helpful in big data 
df_mod['count'] =1

df_mod.groupby(['Type 1']).count()['count']

##Apply multiple parameters 
df_mod.groupby(['Type 1', 'Type 2']).count()['count']
##Read chunk size data when data is too big
##For example read 5 rows at a time

# for df_mod in pd.read_csv('modified_new.csv', chunksize=5):
#     print("Chunk Df")
#     print(df)
## Check number of variable available under numeric_data
numeric_data = df.select_dtypes(exclude = [object])
numeric_data.shape
corr_matrix = numeric_data.corr()
corr_matrix
plt.figure(figsize=(30,20))
sns.heatmap(corr_matrix,annot=True,cmap='YlGnBu')
plt.show()
sns.pairplot(numeric_data, kind='scatter',hue='Legendary')
## Drop Type 2 and Beast columns and chack info
df_clean = df.drop(columns = ['Type 2','Beast'])
df_clean.info()
sns.countplot(x ='Type 1', data=df_clean)
plt.xticks(rotation='vertical')
plt.show()
sns.countplot(x='Legendary', data=df_clean)
plt.show()
sns.boxplot(x='Type 1', y='Total', data = df_clean)
plt.xticks(rotation='vertical')
plt.show()
sns.boxplot(x='Legendary', y='Total', data = df_clean)
plt.show()
## Box whiskers plot & histogram on the same window 
## Split the plotting window into 2 parts

f, (ax_box, ax_hist)= plt.subplots(2, gridspec_kw={"height_ratios": (.15, .85)})
## Add and create  box plot
sns.boxplot(df_clean['Total'], ax=ax_box)

sns.distplot(df_clean["Total"], ax=ax_hist)
plt.show()
##Principal Component Analysis(PCA) used for dimension reduction of dataset

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

%matplotlib inline
pca_col =['HP','Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
scaler = StandardScaler()
scaler.fit(df_mod[pca_col])
scaled_data = scaler.transform(df_mod[pca_col])
np.mean(scaled_data), np.std(scaled_data)
scaled_data
feature_cols = ['feature'+str(i) for i in range(scaled_data.shape[1])]
normalised_pokemon = pd.DataFrame(scaled_data,columns=feature_cols)
normalised_pokemon.tail()
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.6)
principal_components =pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape
principal_comp_df = pd.DataFrame(data = x_pca, columns = ['Principal_comp_1', 'Principal_comp_2'])
principal_comp_df.head(4)
## Make a correlation matrix of PCA dataframe
correlation_mat = principal_comp_df.corr()
correlation_mat
## PLot correlation matrix for better understanding
sns.heatmap(correlation_mat, annot=True, cmap='YlGnBu' )
plt.show()
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
data_labels = df["Name"].copy()
df['Type 1'].value_counts()
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal_comp_1',fontsize=20)
plt.ylabel('Principal_comp_2',fontsize=20)
plt.title("Principal Component Analysis of Pokemon Type 1",fontsize=20)
targets = ['Water','Normal','Grass','Bug','Psychic','Flamer','Electric','Rock','Dragon','Ground','Ghost',        
'Dark', 'Poison', 'Steel','Fighting','Ice','Fairy','Flying']
colors = ['r', 'g', 'y','c','#2c3e50','#ee9ca7','#1565C0','#91EAE4','#654ea3','#ffd89b','#799F0C','#dd1818',
         '#FFF200','#FF8C00','#30E8BF','#603813','#b29f94','#24FE41','#a80077']
for target, color in zip(targets,colors):
    indicesToKeep = df['Type 1'] == target
    plt.scatter(principal_comp_df.loc[indicesToKeep, 'Principal_comp_1']
               , principal_comp_df.loc[indicesToKeep, 'Principal_comp_2'], c = color, s = 50)

plt.legend(targets,prop={'size': 7})
plt.show()