# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/ct-accidental-drug-related-deaths-2012june-2017/Accidental_Drug_Related_Deaths__2012-June_2017.csv'

)

df.head(5)
df.info()
fig = sns.countplot( x = 'Sex', data = df, palette = 'Purples')

fig.set_title('Drug related Deaths for females or males')

plt.show()
plt.figure(figsize=(16,6))

fig = sns.countplot(x='Age', data=df )

plt.xticks(rotation = 45, horizontalalignment = 'right')

plt.show()
sns.countplot( x = 'Race', data = df)

plt.xticks(rotation = 45, horizontalalignment = 'right')

plt.show()
drugs  = []

xlabel = ['Heroin', 'Cocaine', 'Fentanyl', 'Ethanol','Benzodiazepine']



drugs = df.Heroin.count(), df.Cocaine.count(), df.Fentanyl.count(), df.EtOH.count(), df.Benzodiazepine.count()



sns.barplot( x = xlabel, y = drugs, palette = 'Purples')

plt.title('Most frequend drugs used causing Deaths from 2012-2017')

plt.show()
df.head()
df['Year'] = df['Date'].str.slice(start=6) #String slicing for the year

df['Month'] = df['Date'].str.slice(start=0, stop=2) #String slicing for the month

df.head()
order = ['2012','2013','2014','2015','2016','2017'] # sets the Order of the bars

sns.countplot(x=df.Year, data= df, order = order, palette = 'BuGn')

plt.title('Drug related Deaths between 2012-2017')



plt.show()