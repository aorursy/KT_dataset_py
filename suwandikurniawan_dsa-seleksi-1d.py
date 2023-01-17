# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_excel("/kaggle/input/used_car_data.xlsx")
df
df['Location'].groupby(df['Location']).count().sort_values(ascending=False)
import statistics



km_driven = []

category = []



for km in df['Kilometers_Driven']:

    km_driven.append(km)

  

med = statistics.median(km_driven)



for km in km_driven:

    if km >= med:

        category.append("Tinggi")

    else:

        category.append("Rendah")

        

df['Category'] = category
df
df['Owner_Type'].groupby(df['Owner_Type']).count().sort_values(ascending=False)
df[(df['Owner_Type'] == 'Third') | (df['Owner_Type'] == 'Fourth & Above')]['Name'].count()
import matplotlib.pyplot as plt

tahun_raw = [tahun for tahun in df['Year']]

harga_raw = [harga for harga in df['Price']]

tahun = []

harga_rata = []



for year in df['Year']:

    if year not in tahun:

        tahun.append(year)

        tahun.sort()



for thn in tahun:

    harga = 0

    jumlah = 0

    for i in range(len(tahun_raw)):

        if tahun_raw[i] == thn:

            harga+=harga_raw[i]

            jumlah+=1

    harga_rata.append(harga/jumlah)

    harga = 0



df_graph = pd.DataFrame({

   'harga': harga_rata,

   }, index=tahun)

lines = df_graph.plot.line()



fig, ax = plt.subplots(figsize=(16,8))

ax.scatter(tahun, harga_rata)

ax.set_xlabel('Year')

ax.set_ylabel('Average Price')

plt.plot(tahun, harga_rata)

plt.show()

milik_raw = [milik for milik in df['Owner_Type']]

harga_raw = [harga for harga in df['Price']]

milik = []

harga_rata = []



for own in df['Owner_Type']:

    if own not in milik:

        milik.append(own)



for own in milik:

    harga = 0

    jumlah = 0

    for i in range(len(milik_raw)):

        if milik_raw[i] == own:

            harga+=harga_raw[i]

            jumlah+=1

    harga_rata.append(harga/jumlah)

    harga = 0



df_graph = pd.DataFrame({

   'harga': harga_rata,

   }, index=milik)

lines = df_graph.plot.line()