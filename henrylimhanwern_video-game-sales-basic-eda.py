import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.colors



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

%config InlineBackend.figure_format = 'retina'

df.head(5)
df.describe()
fig, axs = plt.subplots(1, 3, sharey=True);

gh1 = df.plot(kind='scatter', x='Year', y='Global_Sales',ax = axs[0], figsize=(22, 6));

gh2 = df.plot(kind='scatter', x='Genre', y='Global_Sales',ax = axs[1], figsize=(22, 6));

gh3 = df.plot(kind='scatter', x='Platform', y='Global_Sales', ax = axs[2], figsize=(22, 6));

gh2.tick_params(labelrotation=45)

gh3.tick_params(labelrotation=90)
GR1 = df.groupby(["Genre","Year"]).sum().reset_index()

GR2 = df.groupby(["Genre","Year"]).size().reset_index(name = 'Frequency')

PR1 = df.groupby(['Platform','Year']).sum().reset_index()

PR2 = df.groupby(['Platform','Year']).size().reset_index(name = 'Frequency')

#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)

fs = plt.figure(figsize=(14,7))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(20,20) )

sns.lineplot(x='Year',y='Global_Sales',  hue = 'Genre', ax=ax1, palette="gist_rainbow",  data=GR1)

sns.lineplot(x='Year',y='Frequency',  hue = 'Genre', ax=ax2, palette="gist_rainbow", data=GR2)

sns.lineplot(x='Year',y='Global_Sales',  hue = 'Platform', ax=ax3, palette="gist_ncar_r",  data=PR1)

sns.lineplot(x='Year',y='Frequency',  hue = 'Platform', ax=ax4, palette="gist_ncar_r",  data=PR2)



PuR1 = df.groupby(["Publisher","Year"]).sum().reset_index()

PuR2 = df.groupby(["Publisher","Year"]).size().reset_index(name = 'Frequency')

Frequency = PuR2['Frequency']

PuR1 = PuR1.join(Frequency)

PuR3 = PuR1.sort_values('Global_Sales',ascending = False)

PuR3 = PuR3.head(100)

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



plt.figure(figsize=(20,13))

ax = sns.scatterplot(x="Year", y="Frequency",hue="Publisher", size="Global_Sales", sizes = (100,800), palette="gist_rainbow",data=PuR3)

SR1 = df.groupby('Year').sum().reset_index()

#SR1 = SR1.melt('Year', var_name=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'],  value_name='vals')

#plt.figure(figsize=(16,9))

#d = sns.factorplot(x="Year", y="vals", hue=['NA_Sales','EU_Sales','JP_Sales','Other_Sales'], data=SR1)

SR1 = pd.melt(SR1, id_vars=['Year'], value_vars=['NA_Sales', 'JP_Sales','EU_Sales','Other_Sales'])

SR1 = SR1.groupby(['variable','Year']).sum().reset_index()

#SR1 = SR1.sort_values('Year',ascending=True)

plt.figure(figsize=(16,9))

ax = sns.lineplot(x='Year',y='value',  hue = 'variable', palette="gist_rainbow",  data=SR1)
