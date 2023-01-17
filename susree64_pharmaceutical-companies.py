import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

%matplotlib inline
df = pd.read_csv("../input/pharmaceuticalcompanies/Pharmaceutical Companies.csv")
df.head()
df.shape
#Make a new data frame without NaN

df1 = df.dropna()

df1.isna().sum()

df1 = df1.drop(['region', 'Image URL'], axis = 1)
df1 = df1.drop(['region', 'Image URL'], axis = 1)

countryNames = list(df1['Country Name'])

#countryNames = ['Year'].append(countryNames)

countryNames.insert(0, 'Year')

countryNames
df1 = df1.iloc[0:, 1:].T.reset_index()

df1.columns = countryNames
fig, axs = plt.subplots(5, figsize = (25, 15))



fig.suptitle("Companies and  Yearly values", fontsize = 25)



axs[0].plot(x, y, linestyle='--', marker='o', color='#348a4b', label = 'Merck' )

axs[0].legend(fontsize = 15)



axs[1].plot(x, y1, linestyle='--', marker='x', color='#256c70', label = 'Novartis')

axs[1].legend(fontsize = 15)



axs[2].plot(x, y2, linestyle='--', marker='*', color='#705325', label = 'Johnson & Johnson')

axs[2].legend(fontsize = 15)



axs[3].plot(x, y3, linestyle='--', marker='*', color='#7c6191', label = 'Roche')

axs[3].legend(fontsize = 15)



axs[4].plot(x, y4, linestyle='--', marker='x', color='#8a86ba', label = 'Pfizer')

axs[4].legend(fontsize = 15)

plt.xlabel("Years", fontsize = 30)







plt.show()