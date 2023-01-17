import pandas as pd

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np





maindata = pd.read_csv('../input/main_data.csv') 



#Fill in any null data

maindata['AoA'].fillna(maindata['AoA'].mean(),inplace=True)

maindata['VSoA'].fillna(maindata['VSoA'].mean(),inplace=True)

maindata['Freq'].fillna(maindata['Freq'].mean(),inplace=True)

maindata['CDS_freq'].fillna(maindata['CDS_freq'].mean(),inplace=True)

maindata['Lex_cat'].fillna('unknown',inplace=True)

maindata['Broad_lex'].fillna('unknown',inplace=True)
print(maindata.head(5))
sns.lmplot('VSoA','AoA',data=maindata, fit_reg=True)

sns.plt.title('Typical age to learn word vs how many other words are known at this age')

sns.plt.show()
sns.lmplot('Freq','AoA',data=maindata, fit_reg=True)

sns.plt.title('Typical age to learn word vs word frequency in Norwegian')

sns.plt.show()
sns.lmplot('CDS_freq','AoA',data=maindata, fit_reg=True)

sns.plt.title('Typical age to learn word vs how often this word is used when talking to children')

sns.plt.show()
sns.boxplot(maindata['Broad_lex'],maindata['AoA'])

sns.plt.xticks(rotation=45)

sns.plt.show()

print(maindata[maindata['Broad_lex']=='games & routines']['Translation'].head(10))
print(maindata[maindata['Broad_lex']=='nominals']['Translation'].head(10))
print(maindata[maindata['Broad_lex']=='closed-class']['Translation'].head(10))
#Convert categorical data to numeric values

var_mod = ['Broad_lex','Lex_cat']

le = LabelEncoder()

for i in var_mod:

	maindata[i] = le.fit_transform(maindata[i])

temp = ['AoA','VSoA','Lex_cat','Broad_lex','Freq','CDS_freq']

corr = maindata[temp].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



sns.heatmap(corr, mask=mask,linewidths=.5,cbar_kws={"shrink":.5})

print(corr)