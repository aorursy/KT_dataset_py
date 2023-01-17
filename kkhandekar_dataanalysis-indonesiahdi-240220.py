#Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

data = pd.read_csv('../input/indonesias-human-development-index-20102014/Indeks-Pembangunan-Manusia-Menurut-Provinsi-di-Indonesia-2010-2014.csv')
data = data.rename(columns={"tahun":"Year","provinsi":"Province","indeks":"Index"})
data.head()
data.isna().sum()
data.shape
data.groupby('Province').size()
plt.style.use('seaborn-deep')

plt.figure(figsize=(10,10))

plt.grid(True)

x = data['Index']

plt.hist([x], label=['Index Distribution'])

plt.legend(loc='upper right')

plt.title('HDI Distribution')

plt.show()
#creating a seperate dataset of top 8 province (based on population)

data_mini = data[data['Province'].isin(['Bali','Jawa Tengah','Jawa Timur','Jawa Barat','Sumatera Utara','DKI Jakarta','Banten','Sulawesi Selatan'])]
data_mini.head()
sns.set(style="ticks")

fig = plt.figure()

fig = sns.relplot(x="Year", y="Index", kind="line", hue="Province",data=data_mini,size='Province')

fig.fig.set_size_inches(15,10)

fig.set(xticks=data_mini.Year)

fig.set_titles("HDI (2010-2014)")

fig.set_xlabels("Year")

fig.set_ylabels("Index")

plt.show()
sns.set(style="ticks")



# Initialize the figure with a logarithmic x axis

f, ax = plt.subplots(figsize=(15, 10))



# Plot the orbital period with horizontal boxes

sns.boxplot(x="Index", y="Province", data=data_mini,

            palette="vlag")



# Tweak the visual presentation

ax.xaxis.grid(True)

ax.set(ylabel="")

sns.despine(trim=True, left=True)
def index_classify(ind):

    index_cat = ['Medium','High']

    if 55.0 <= ind <= 69.9:

        return index_cat[0]

    elif 70.0 <= ind <= 79.9:

        return index_cat[1]

    else:

        return 'Out of Bounds'



#Applying the function

data_mini['index_category'] = data_mini['Index'].apply(index_classify)
data_mini.head()
sns.pairplot(data_mini, hue='index_category')