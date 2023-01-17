import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_2016=pd.read_csv('/kaggle/input/SampleResults 2016.csv')
data_2016.head()
data_2016=data_2016.rename(columns={'Pesticide Name':'Pesticide_Name'}) 
data_2016['Commodity'].value_counts()
data_2016_usa=data_2016.loc[data_2016.Origin.isin(['1'])]

data_2016_usa.isnull().sum() 
data_2016_usa = data_2016_usa[data_2016_usa['State'].notna()]

data_2016_usa.isnull().sum() 
data_2016_usa.groupby('Commodity')['Pesticide_Name'].describe()
data_2016_usa.groupby('Commodity')['State'].describe()
data_2016_usa.groupby('State')['Pesticide_Name'].describe()
apple_data_usa=data_2016_usa.loc[data_2016_usa.Commodity=='AP',:] 

apple_data_usa['Variety'].value_counts(normalize=True)
apple_data_usa=apple_data_usa.loc[apple_data_usa.Variety.isin(['Red Delicious', 'Granny Smith', 'Gala','Fuji','Golden Delicious'])]

pd.options.display.max_columns = None

pd.options.display.max_rows = None
apple_data_usa.groupby('Variety')['Pesticide_Name'].value_counts(normalize=True)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'DPA', 'Thiabendazole', 'Fludioxonil', 'Chlorantraniliprole', 'Acetamiprid','Pyrimethanil','Other'

ax1.sizes = [19.7, 16.6, 9.5, 7.4, 6.7, 6.7, 33.4]

ax1.set_title("Pesticides in Domestic Fuji Apple Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90, radius=4)

ax1.axis('equal')



ax2.labels='DPA', 'Thiabendazole', 'Chlorantraniliprole', 'Boscalid', 'Pyrimethanil', 'Pyraclostrobin', 'Other'

ax2.sizes=[16.7,15.4,8.5,8.2,7.8,7.5,35.9]

ax2.set_title("Pesticides in Domestic Gala Apple Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(15,5)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'DPA', 'Thiabendazole', 'Fludioxonil', 'Pyrimethanil','Acetamiprid','Boscalid','Other'

ax1.sizes = [22.2, 20, 9.5, 8.3, 7.7, 6.1, 26.2]

ax1.set_title("Pesticides in Domestic Granny Smith Apple Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')



ax2.labels = 'DPA', 'Thiabendazole', 'Fludioxonil', 'Pyrimethanil','Acetamiprid','Chlorantraniliprole','Spirodiclofen','Other'

ax2.sizes = [20, 13.6, 8.8, 8.8, 6.1, 6, 4.4, 32.3]

ax2.set_title("Pesticides in Domestic Red Delicious Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax) = plt.subplots()

ax.labels = 'DPA', 'Thiabendazole', 'Fludioxonil', 'Acetamiprid','Boscalid','Chlorantraniliprole','Pyraclostrobin','Other'

ax.sizes = [19.3, 12.9, 8, 6.4, 6.4, 6.4, 5.6, 35]

ax.set_title("Pesticides in Domestic Golden Delicious Apple Samples")

ax.pie(ax.sizes, labels=ax.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')

fig = plt.gcf()

fig.set_size_inches(12,4)
apple_data_usa.groupby('Variety')['State'].describe()
apple_data_usa.groupby('State')['Pesticide_Name'].describe()
a1=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Diphenylamine (DPA)',:] 

a1[a1['Concentration']>=10.0]
sns.distplot(a1['Concentration'])
a2=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Thiabendazole',:] 

a2[a2['Concentration']>=5.0]
sns.distplot(a2['Concentration'])
a3=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Fludioxonil',:] 

a3[a3['Concentration']>=5.0]
sns.distplot(a3['Concentration'])
a4=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Chlorantraniliprole',:] 

a4[a4['Concentration']>=1.2]
sns.distplot(a4['Concentration'])
a5=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Acetamiprid',:] 

a5[a5['Concentration']>=1.0]
sns.distplot(a5['Concentration'])
a6=apple_data_usa.loc[apple_data_usa.Pesticide_Name=='Boscalid',:] 

a6[a6['Concentration']>=3.0]
sns.distplot(a6['Concentration'])
data_2016_imports=data_2016.loc[data_2016.Origin.isin(['2'])]

apple_data_imports=data_2016_imports.loc[data_2016_imports.Commodity=='AP',:] 

apple_data_imports['Variety'].value_counts(normalize=True)
apple_data_imports=apple_data_imports.loc[apple_data_imports.Variety.isin(['Gala', 'Royal Gala', 'Fuji'])]

apple_data_imports.groupby('Variety')['Pesticide_Name'].value_counts(normalize=True)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Acetamiprid', 'Methoxyfenozide', 'Thiabendazole', 'Carbendazim (MBC)', 'Chlorantraniliprole','Pyrimethanil','Other'

ax1.sizes = [23, 15.3, 15.3, 7.6, 7.6, 7.6, 23.6]

ax1.set_title("Pesticides in Imported Fuji Apple Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90, radius=4)

ax1.axis('equal')



ax2.labels='Acetamiprid', 'Pyrimethanil', 'Thiacloprid', 'Chlorantraniliprole', 'Spirodiclofen', 'Other'

ax2.sizes=[22.7,22.7,18.2,9.1,9.1,18.2]

ax2.set_title("Pesticides in Imported Gala Apple Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax) = plt.subplots()

ax.labels = 'Acetamiprid', 'Methoxyfenozide', 'Tetrahydrophthalimide (THPI)', 'Pyrimethanil','Chlorantraniliprole','Other'

ax.sizes = [21.4, 21.4, 21.4, 14.2, 7.1, 14.5]

ax.set_title("Pesticides in Imported Royal Gala  Apple Samples")

ax.pie(ax.sizes, labels=ax.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')

fig = plt.gcf()

fig.set_size_inches(12,4)
apple_data_imports.groupby('Country')['Pesticide_Name'].value_counts(normalize=True)
a7=apple_data_imports.loc[apple_data_imports.Pesticide_Name=='Acetamiprid',:] 

a7[a7['Concentration']>=1.0]
sns.distplot(a7['Concentration'])
a8=apple_data_imports.loc[apple_data_imports.Pesticide_Name=='Methoxyfenozide',:] 

a8[a8['Concentration']>=2.0]
sns.distplot(a8['Concentration'])
a9=apple_data_imports.loc[apple_data_imports.Pesticide_Name=='Pyrimethanil',:] 

a9[a9['Concentration']>=15.0]
sns.distplot(a9['Concentration'])
grape_data_usa=data_2016_usa.loc[data_2016_usa.Commodity=='GR',:] 

grape_data_usa.groupby('State')['Pesticide_Name'].value_counts()
grape_data_imports=data_2016_imports.loc[data_2016_imports.Commodity=='GR',:] 

grape_data_imports['Country'].value_counts()
grape_data_imports=grape_data_imports[grape_data_imports.Country != 'Unknown Country']

grape_data_imports['Country'].value_counts()
grape_data_imports['Variety'].value_counts()
grape_data_imports=grape_data_imports.loc[grape_data_imports.Variety.isin(['Crimson Seedless', 'Thompson Seedless', 'Red Seedless','Flame Seedless','Green Seedless'])]

grape_data_imports.groupby('Variety')['Pesticide_Name'].value_counts(normalize=True)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Boscalid', 'Cyprodinil', 'Fenhaxamid', 'Tebuconazole', 'Difenoconazole','Fludioxonil',\

'Pyrimethanil','Other'

ax1.sizes = [13.6, 11.4, 11.4, 9.2, 7.3, 5.3, 5, 36.8]

ax1.set_title("Pesticides in Imported Crimson Seedless Grape Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90, radius=4)

ax1.axis('equal')



ax2.labels='Tebuconazole', 'Boscalid', 'Cyprodinil', 'Myclobutanil', 'Pyraclostrobin', 'Quinoxyfen',\

'Difenoconazole','Fenhexamid','Trifloxystrobin','Other'

ax2.sizes=[15.2,10.1,7.3,6.7,6.7,6.7,6.2,6.2,5.1,29.8]

ax2.set_title("Pesticides in Imported Flame Seedless")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Tebuconazole', 'Boscalid', 'Fenhexamid', 'Quinoxyfen','Trifloxystrobin','Cyprodinil',\

'Myclobutanil','Iprodione','Other'

ax1.sizes = [14.7, 11.3, 8.3, 7.8, 6.8, 6.4, 5.9, 5.4, 33.4]

ax1.set_title("Pesticides in Imported Red Seedless Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')



ax2.labels = 'Fenhexamid', 'Boscalid', 'Cyprodinil', 'Tebuconazole','Fludioxonil','Pyrimethanil',\

'Trifloxystrobin','Other'

ax2.sizes = [13.5, 11.7, 10, 9.6, 8.2, 6.1, 5.2, 35.7]

ax2.set_title("Pesticides in Imported Thompson Seedless Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax) = plt.subplots()

ax.labels = 'Boscalid', 'Cyprodinil','Tebuconazole','Fenhaxamid','Trifloxystrobin','Fludioxonil','Iprodione',\

'Quinoxyfen','Other'

ax.sizes = [11.2, 10.6, 10.6, 9.4, 6.5, 5.9, 5.9, 5.3, 34.6]

ax.set_title("Pesticides in Imported Green Seedless Grape Samples")

ax.pie(ax.sizes, labels=ax.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')

fig = plt.gcf()

fig.set_size_inches(12,4)
grape_data_imports.groupby('Variety')['Country'].describe()
grape_data_imports.groupby('Country')['Pesticide_Name'].describe()
g1=grape_data_imports.loc[grape_data_imports.Pesticide_Name=='Boscalid',:] 

g1[g1['Concentration']>=5.0]
sns.distplot(g1['Concentration'])
g2=grape_data_imports.loc[grape_data_imports.Pesticide_Name=='Tebuconazole',:]

g2[g2['Concentration']>=5.0]
sns.distplot(g2['Concentration'])
g3=grape_data_imports.loc[grape_data_imports.Pesticide_Name=='Cyprodinil',:]

g3[g3['Concentration']>=3.0]
sns.distplot(g3['Concentration'])
g4=grape_data_imports.loc[grape_data_imports.Pesticide_Name=='Fenhexamid',:] 

g4[g4['Concentration']>=4.0]
sns.distplot(g4['Concentration'])
grape_data_usa['Variety'].value_counts(normalize=True)
grape_data_usa=grape_data_usa.loc[grape_data_usa.Variety.isin(['Red Seedless', 'Green Seedless', 'Flame Seedless','Thompson Seedless','Crimson Seedless'])]

grape_data_usa.groupby('Variety')['Pesticide_Name'].value_counts(normalize=True)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Boscalid', 'Pyraclostrobin','Chlorantraniliprole','Quinoxyfen','Buprofezin',\

'Cyprodinil','Etoxazole','Fenpropathrin','Myclobutanil','Other'

ax1.sizes = [16.6, 16.6, 8.3, 8.3, 5.5, 5.5, 5.5, 5.5, 5.5, 31]

ax1.set_title("Pesticides in Domestic Crimson Seedless Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90, radius=4)

ax1.axis('equal')



ax2.labels='Boscalid', 'Pyraclostrobin', 'Buprofezin', 'Trifloxystrobin', 'Quinoxyfen', 'Myclobutanil',\

'Cyprodinil','Other'

ax2.sizes=[16.1,14.6,9.7,9.7,8.4,7.6, 6.9, 27.1]

ax2.set_title("Pesticides in Domestic Flame Seedless Grape Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')





fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Boscalid', 'Pyraclostrobin', 'Spirotetramat', 'Cyprodinil','Myclobutanil','Quinoxyfen',\

'Tebuconazole','Fenhexamid','Tetraconazole', 'Other'

ax1.sizes = [11.1,9.4, 9, 8.3, 6, 5.3, 5.3, 4, 4, 37.6]

ax1.set_title("Pesticides in Domestic Red Seedless Grape Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')



ax2.labels = 'Boscalid', 'Buprofezin', 'Pyraclostrobin','Chlorantraniliprole','Quinoxyfen',\

'Trifloxystrobin','Other'

ax2.sizes = [15, 15, 15, 10, 10, 10, 25]

ax2.set_title("Pesticides in Domestic Thompson Seedless Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax) = plt.subplots()

ax.labels = 'Cyprodinil', 'Boscalid', 'Pyraclostrobin', 'Quinoxyfen','Trifloxystrobin','Myclobutanil',\

'Spirotetramat','Etoxazole', 'Other'

ax.sizes = [11, 10, 9.1, 6.8, 6.8, 6.3, 5.4, 5, 39.6]

ax.set_title("Pesticides in Domestic Green Seedless Grape Samples")

ax.pie(ax.sizes, labels=ax.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax.axis('equal')

fig = plt.gcf()

fig.set_size_inches(12,4)
grape_data_imports.groupby('Variety')['Pesticide_Name'].describe()
grape_data_usa.groupby('Variety')['Pesticide_Name'].describe()
g5=grape_data_usa.loc[grape_data_usa.Pesticide_Name=='Boscalid',:] 

g5[g5['Concentration']>=5.0]
sns.distplot(g5['Concentration'])
g6=grape_data_usa.loc[grape_data_usa.Pesticide_Name=='Cyprodinil',:] 

g6[g6['Concentration']>=3.0]
sns.distplot(g6['Concentration'])
g7=grape_data_usa.loc[grape_data_usa.Pesticide_Name=='Pyraclostrobin',:] 

g7[g7['Concentration']>=2.0]
sns.distplot(g7['Concentration'])
g8=grape_data_usa.loc[grape_data_usa.Pesticide_Name=='Quinoxyfen',:] 

g8[g8['Concentration']>=2.0]
sns.distplot(g8['Concentration'])
g9=grape_data_usa.loc[grape_data_usa.Pesticide_Name=='Spirotetramat',:] 

g9[g9['Concentration']>=1.3]
sns.distplot(g9['Concentration'])
lettuce_data_usa=data_2016_usa.loc[data_2016_usa.Commodity=='LT',:]

lettuce_data_usa['Variety'].value_counts()
lettuce_data_usa=lettuce_data_usa.loc[lettuce_data_usa.Variety.isin(['Iceberg', 'Romaine', 'Green Leaf','Red Leaf'])]

lettuce_data_usa.groupby('Variety')['Pesticide_Name'].value_counts(normalize=True)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Imidacloprid', 'Cyhalothrin', 'Mandipropamid', 'Spinetoram','DCPA','Propamocarb hydrochloride',\

'Dimethomorph','Fenamidone','Permethrin cis', 'Other'

ax1.sizes = [10.3, 7.1, 6.8, 6.5, 6.3, 6, 5.4, 5.4, 5.1, 41.1]

ax1.set_title("Pesticides in Domestic Green Leaf Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')



ax2.labels = 'Imidacloprid','Propamocarb hydrochloride','Thiamethoxam','Dimethomorph','Fenamidone','Acephate',\

'Mandipropamid','Other'

ax2.sizes = [25.2, 14.1, 13.1, 9.2, 6.5, 5.2, 5.0 ,21.7]

ax2.set_title("Pesticides in Domestic Iceberg Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.labels = 'Imidacloprid','Fenamidone','Mandipropamid','Cypermethrin','Permethrin cis','Permethrin trans',\

'Spinetoram','Other'

ax1.sizes = [12.5, 11.1, 6.9, 5.5, 5.5, 5.5, 5.5, 52.5]

ax1.set_title("Pesticides in Domestic Red Leaf Samples")

ax1.pie(ax1.sizes, labels=ax1.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')



ax2.labels = 'Imidacloprid', 'Mandipropamid', 'Cyhalothrin', 'Spinetoram','Permethrin trans','Permethrin cis',\

'Other'

ax2.sizes = [8.9, 8.2, 7.2, 6.5, 6.0, 5.8,57.4]

ax2.set_title("Pesticides in Domestic Romaine Samples")

ax2.pie(ax2.sizes, labels=ax2.labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax2.axis('equal')



fig = plt.gcf()

fig.set_size_inches(12,4)
lettuce_data_usa.groupby('State')['Pesticide_Name'].describe()
grape_data_usa.groupby('State')['Pesticide_Name'].describe()
l1=lettuce_data_usa.loc[lettuce_data_usa.Pesticide_Name=='Imidacloprid',:] 

l1[l1['Concentration']>=3.5]
sns.distplot(l1['Concentration'])
l2=lettuce_data_usa.loc[lettuce_data_usa.Pesticide_Name=='Cyhalothrin, Total (Cyhalothrin-L + R157836 epimer)',:] 

l2[l2['Concentration']>=2.0]
sns.distplot(l2['Concentration'])
l3=lettuce_data_usa.loc[lettuce_data_usa.Pesticide_Name=='Mandipropamid',:] 

l3[l3['Concentration']>=20]
sns.distplot(l3['Concentration'])
l4=lettuce_data_usa.loc[lettuce_data_usa.Pesticide_Name=='Fenamidone',:] 

l4[l4['Concentration']>=60]
sns.distplot(l4['Concentration'])
lettuce_data_imports=data_2016_imports.loc[data_2016_imports.Commodity=='LT',:] 

lettuce_data_imports['Country'].value_counts()
lettuce_data_imports['Variety'].value_counts()
lettuce_data_imports
lettuce_data_imports=lettuce_data_imports.loc[lettuce_data_imports.Variety.isin(['Romaine'])]

lettuce_data_imports.groupby('Variety')['Pesticide_Name'].value_counts()